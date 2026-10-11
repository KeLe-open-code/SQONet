"""Count errors and micro localization precision/recall/F1. Hungarian assignment on full Euclidean distances, then a radius threshold on the 800-pixel canvas."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import contracts as C


def hungarian_pairs(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Matched (i, j, d) triples from one optimal assignment."""
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((0, 3), np.float64)
    dist = np.linalg.norm(
        np.asarray(pred, np.float64)[:, None, :]
        - np.asarray(gt, np.float64)[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(dist)
    if len(rows) == 0:
        return np.zeros((0, 3), np.float64)
    d = dist[rows, cols]
    return np.stack([rows, cols, d], axis=1)


def match_tp(pred: np.ndarray, gt: np.ndarray, radius: float) -> int:
    pairs = hungarian_pairs(pred, gt)
    return int((pairs[:, 2] <= radius).sum()) if len(pairs) else 0


def evaluate_rows(rows: Sequence[dict], point_model: bool = True) -> dict:
    """rows: [{key, gt, pred, pred_xy?, gt_xy?, scene?, density_count?}, ...]

    For point models pred is len(final point set) and pred_xy is the (N,2)
    array; for density models pred is the native integral and F1 is None."""
    out: dict = {}
    n = len(rows)
    if n == 0:
        return out
    ae = np.asarray([abs(float(r["pred"]) - float(r["gt"])) for r in rows], np.float64)
    se = np.asarray([(float(r["pred"]) - float(r["gt"])) ** 2 for r in rows], np.float64)
    base = dict(
        samples=n,
        mae=float(ae.mean()),
        rmse=float(np.sqrt(se.mean())),
        mean_error=float(np.mean([float(r["pred"]) - float(r["gt"]) for r in rows])),
        avg_pred=float(np.mean([float(r["pred"]) for r in rows])),
        avg_gt=float(np.mean([float(r["gt"]) for r in rows])),
        total_pred=int(sum(int(r["pred"]) if point_model else 0 for r in rows)) if point_model else float(sum(float(r["pred"]) for r in rows)),
        total_gt=int(sum(int(r["gt"]) for r in rows)),
    )
    if not point_model:
        base.update(f1=None, precision=None, recall=None)
        return base

    tot_pred = sum(int(r["pred"]) for r in rows)
    tot_gt = sum(int(r["gt"]) for r in rows)
    per_r: dict[int, dict] = {}
    macro_f1_10: list[float] = []
    hit_10 = 0
    for r_px in C.MATCH_RADII:
        tp = fp = fn = 0
        for r in rows:
            pxy = np.asarray(r["pred_xy"] if r.get("pred_xy") is not None
                             else np.zeros((0, 2)), np.float64).reshape(-1, 2)
            gxy = np.asarray(r["gt_xy"] if r.get("gt_xy") is not None
                             else np.zeros((0, 2)), np.float64).reshape(-1, 2)
            pairs = hungarian_pairs(pxy, gxy)
            t = int((pairs[:, 2] <= r_px).sum()) if len(pairs) else 0
            tp += t
            fp += int(len(pxy) - t)
            fn += int(len(gxy) - t)
            if r_px == C.PRIMARY_RADIUS:
                p = t / max(len(pxy), 1)
                rc = t / max(len(gxy), 1)
                macro_f1_10.append(2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0)
                hit_10 += int(t > 0)
        prec = tp / tot_pred if tot_pred else (1.0 if tot_gt == 0 else 0.0)
        rec = tp / tot_gt if tot_gt else (1.0 if tot_pred == 0 else 0.0)
        f1 = (2 * tp / (tot_pred + tot_gt)) if (tot_pred + tot_gt) else 1.0
        per_r[r_px] = dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1,
                           all_empty=(tot_pred == 0 and tot_gt == 0))
    m = per_r[C.PRIMARY_RADIUS]
    base.update(
        tp=m["tp"], fp=m["fp"], fn=m["fn"],
        precision=m["precision"], recall=m["recall"], f1=m["f1"],
        micro_f1_by_radius={str(r): v["f1"] for r, v in per_r.items()},
        tp_by_radius={str(r): v["tp"] for r, v in per_r.items()},
        macro_f1_10=float(np.mean(macro_f1_10)) if macro_f1_10 else None,
        hit_image_rate_10=(hit_10 / n) if n else None,
    )
    return base


def aggregate_by_scene(rows: Sequence[dict], point_model: bool = True) -> dict:
    scenes = sorted({r.get("scene", "") for r in rows})
    out = {"overall": evaluate_rows(rows, point_model)}
    for s in scenes:
        if not s:
            continue
        out[f"scene_{s}"] = evaluate_rows([r for r in rows if r.get("scene") == s], point_model)
    # density bins (diagnostic)
    for name, lo, hi in [("0_25", 0, 25), ("26_50", 26, 50), ("51_100", 51, 100),
                         ("101_200", 101, 200), ("201_plus", 201, 1 << 30)]:
        sel = [r for r in rows if lo <= int(r["gt"]) <= hi]
        if sel:
            sub = evaluate_rows(sel, point_model)
            sub.pop("micro_f1_by_radius", None)
            sub.pop("tp_by_radius", None)
            out[f"density_{name}"] = dict(samples=len(sel), mae=sub["mae"], rmse=sub["rmse"])
    return out


def make_row(key: str, gt_count: int, pred_count, 
             pred_xy=None, gt_xy=None, scene: str = "") -> dict:
    """pred_count: int for point models (final set size), float for density."""
    return dict(key=key, gt=int(gt_count), pred=pred_count,
                pred_xy=np.asarray(pred_xy, np.float32).reshape(-1, 2) if pred_xy is not None else None,
                gt_xy=np.asarray(gt_xy, np.float32).reshape(-1, 2) if gt_xy is not None else None,
                scene=scene)


def rows_to_jsonl(rows: Sequence[dict]) -> list[dict]:
    """Serialization form: coords as lists, drop None coords."""
    out = []
    for r in rows:
        rr = dict(key=r["key"], gt=r["gt"], pred=r["pred"],
                  ae=abs(r["pred"] - r["gt"]), scene=r.get("scene", ""))
        if r.get("pred_xy") is not None:
            rr["pred_xy"] = np.asarray(r["pred_xy"], np.float32).round(4).tolist()
        if r.get("gt_xy") is not None:
            rr["gt_xy"] = np.asarray(r["gt_xy"], np.float32).round(4).tolist()
        out.append(rr)
    return out
