"""Research constants with portable public-release paths."""
from pathlib import Path
import hashlib
import json
import os
ROOT = Path(__file__).resolve().parents[1]
P2P_SRC = ROOT / "third_party" / "p2pnet"
DATA_ROOT = Path(os.environ.get("SQONET_DB_ROOT", str(ROOT / "data" / "DB")))
IMAGE_ROOT = Path(os.environ.get("SQONET_IMAGE_ROOT", str(DATA_ROOT / "images")))
DINO_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_VENDOR = ROOT / "third_party"
DINO_WEIGHT = Path(os.environ.get("SQONET_DINO_WEIGHTS", str(ROOT / "weights" / "dinov2_vits14_pretrain.pth")))
DINO_SIZE = 798
DINO_TOKENS = 57
DINO_DIM = 384
SAF_GRID = 100
SEEDS = [42, 3407, 2026]
SIZE = 800
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MATCH_RADII = (5, 10, 15, 20)
PRIMARY_RADIUS = 10
MIN_DELTA = 0.02       # anchor improvement threshold (patience semantics)
PATIENCE = 5           # consecutive stale epochs after min_epoch
EFFECTIVE_BATCH = 8
MICRO_BATCH = 2

# P2P family trains in fp32 (PAPER_PROTOCOL section 4); no AMP anywhere in
# the family stages.
FP32 = True

# Stage recipes: min/max epochs, LR scheme, flip augmentation.
# source/target P2P stages use StepLR drop; module stages have no scheduler.
STAGE_RECIPES = {
    "p2p_source": dict(min_epoch=20, max_epoch=60, lr_head=1e-4, lr_backbone=1e-5,
                       weight_decay=1e-4, clip=0.1, scheduler="step30", flip=True,
                       optimizer="adam", budget_hours=24.0),
    "p2p_target": dict(min_epoch=40, max_epoch=80, lr_head=1e-4, lr_backbone=1e-5,
                       weight_decay=1e-4, clip=0.1, scheduler="step40", flip=True,
                       optimizer="adam", budget_hours=8.0),
    "bqs": dict(min_epoch=30, max_epoch=60, lr=1e-3, weight_decay=1e-4, clip=1.0,
                scheduler=None, flip=True, optimizer="adamw", budget_hours=8.0),
    "owls": dict(min_epoch=10, max_epoch=30, lr=3e-4, weight_decay=1e-4, clip=1.0,
                 scheduler=None, flip=True, optimizer="adamw", budget_hours=8.0),
    "qdpr": dict(min_epoch=1, max_epoch=12, lr_old=1e-5, lr_new=1e-4,
                 weight_decay=1e-4, clip=1.0, scheduler=None, flip=False,
                 optimizer="adamw_split", budget_hours=8.0),
    "saf": dict(min_epoch=1, max_epoch=10, lr=1e-3, weight_decay=1e-4, clip=1.0,
                scheduler=None, flip=False, optimizer="adamw", budget_hours=8.0),
}
# point_loss_coef follows the certified B* chain (rqd2/matcher family), which
# is 0.0002, not the older 0.0001 baseline CFG; both appear in the source
# trees so the choice is recorded explicitly here. The PLC env override exists
# for the sensitivity study only; unset keeps the frozen paper value.
import os as _os
POINT_LOSS_COEF = float(_os.environ.get("PLC", "0.0002"))
EOS_COEF = 0.5
SET_COST_CLASS = 1.0
SET_COST_POINT = 0.05


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    import os
    os.replace(tmp, path)


