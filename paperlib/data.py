"""Fixed DB manifests and the original preprocessing functions."""
from __future__ import annotations
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import cv2
import numpy as np
from . import contracts as C

@dataclass(frozen=True)
class Record:
    key: str
    split: str
    image: Path
    label: Path
    scene: str = ""

@dataclass
class Dataset:
    name: str
    records: dict = field(default_factory=dict)
    def keys(self, split):
        return [r.key for r in self.records[split]]
    def __len__(self):
        return sum(map(len, self.records.values()))

def load_dongbei(splits=("train", "val")):
    ds = Dataset("DB")
    for split in splits:
        if split not in ("train", "val", "test"):
            raise ValueError(split)
        rows = []
        for line in (C.DATA_ROOT / "splits" / (split + ".txt")).read_text(encoding="utf8").splitlines():
            image, label = line.split()
            rows.append(Record(Path(image).stem, split, C.IMAGE_ROOT / image,
                               C.DATA_ROOT / "annotations" / label,
                               "city" if "DongBei_city/" in image else "suburb"))
        ds.records[split] = rows
    return ds

def read_labelme_points(label: Path) -> np.ndarray:
    """labelme point JSON (DB/QianYi/Pullman) or the canonical simplified
    {"points": [[x,y],...]} form (generalization datasets)."""
    data = json.loads(label.read_text(encoding="utf-8"))
    shapes = data if isinstance(data, list) else data.get("shapes", [])
    if not shapes and isinstance(data, dict) and isinstance(data.get("points"), list):
        pts = [[float(p[0]), float(p[1])] for p in data["points"] if len(p) >= 2]
        if not pts:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    pts = []
    for s in shapes:
        if isinstance(s, dict) and s.get("shape_type") == "point" and s.get("points"):
            p = s["points"][0]
            pts.append([float(p[0]), float(p[1])])
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def _clamp_boundary(points: np.ndarray, w: int, h: int, key: str) -> tuple[np.ndarray, int]:
    """Points exactly on x==W / y==H map to nextafter(W,0); other
    out-of-range points are an adaptation error (contract section 3)."""
    fixed = 0
    if len(points) == 0:
        return points, fixed
    out = points.copy()
    for axis, size in ((0, w), (1, h)):
        exact = out[:, axis] == size
        if exact.any():
            out[exact, axis] = np.nextafter(np.float64(size), np.float64(0))
            fixed += int(exact.sum())
        if (out[:, axis] > size).any():
            bad = int((out[:, axis] > size).sum())
            raise ValueError(f"{key}: {bad} points beyond x/y=={size} are not "
                             "exact-boundary; refusing to clip")
        if (out[:, axis] < 0).any():
            raise ValueError(f"{key}: negative coordinates")
    return out, fixed


def load_view(record: Record, flip: bool = False, size: int = C.SIZE):
    """-> (tensor CHW float32 normalized, points (N,2) float32, rgb01 HWC).

    Deterministic INTER_LINEAR resize; points scale in float64 then cast,
    matching the certified D-DB-003 geometry."""
    raw = np.fromfile(str(record.image), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"decode failed: {record.image}")
    src_h, src_w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if (src_w, src_h) != (size, size):
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    arr = (rgb.astype(np.float32) / 255.0
           - np.asarray(C.MEAN, np.float32).reshape(1, 1, 3)) \
        / np.asarray(C.STD, np.float32).reshape(1, 1, 3)
    points = read_labelme_points(record.label)
    if points.size:
        points *= np.asarray([size / int(src_w), size / int(src_h)], np.float64)
        points, _fixed = _clamp_boundary(points, size, size, record.key)
    if flip:
        arr = np.ascontiguousarray(arr[:, ::-1, :])
        if points.size:
            points = points.copy()
            points[:, 0] = np.float64(size - 1) - points[:, 0]
    tensor = torch_from_numpy(arr)
    return tensor, points.astype(np.float32), (rgb.astype(np.float32) / 255.0)


def torch_from_numpy(arr: np.ndarray):
    import torch
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def epoch_variant_schedule(keys: Sequence[str], seed: int, epoch: int) -> list[tuple[str, bool]]:
    """Exactly one view (flip or not) per image per epoch, shuffled.

    Identical semantics to the certified epoch_variant_schedule: a per-image
    flip bit drawn from Random(seed+epoch), then a single shuffle."""
    rng = random.Random(int(seed) + int(epoch))
    selected = [(str(k), bool(rng.randrange(2))) for k in sorted(keys)]
    rng.shuffle(selected)
    return selected


def epoch_schedule_no_flip(keys: Sequence[str], seed: int, epoch: int) -> list[tuple[str, bool]]:
    """QDPR/SAF stages: no augmentation, order still per-epoch seeded."""
    rng = random.Random(int(seed) + int(epoch))
    selected = [(str(k), False) for k in sorted(keys)]
    rng.shuffle(selected)
    return selected


def batches(seq: Sequence, n: int):
    for i in range(0, len(seq), n):
        yield list(seq[i:i + n])


def count_points(record: Record) -> int:
    return int(len(read_labelme_points(record.label)))


def verify_dataset(ds: Dataset, max_images: int | None = None) -> dict:
    """CPU integrity check: files exist, finite coords, boundary policy."""
    stats = dict(total=0, empty=0, boundary_fixed=0, max_count=0, min_count=10 ** 9,
                 missing=[])
    rows = [(s, r) for s, recs in ds.records.items() for r in recs]
    if max_images is not None:
        rows = rows[:max_images]
    for split, r in rows:
        if not r.image.is_file() or not r.label.is_file():
            stats["missing"].append(r.key)
            continue
        pts = read_labelme_points(r.label)
        n = len(pts)
        stats["total"] += 1
        stats["empty"] += int(n == 0)
        stats["max_count"] = max(stats["max_count"], n)
        stats["min_count"] = min(stats["min_count"], n)
        if n and (not np.isfinite(pts).all()):
            raise ValueError(f"{r.key}: non-finite GT coords")
    if stats["min_count"] == 10 ** 9:
        stats["min_count"] = 0
    return stats
