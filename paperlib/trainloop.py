"""Validation-based selection and resumable training from the research implementation."""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from . import contracts as C


def rng_state() -> dict:
    st = dict(python=random.getstate(), numpy=np.random.get_state(),
              torch=torch.get_rng_state())
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if torch.cuda.is_available() and st.get("cuda") is not None:
        torch.cuda.set_rng_state_all(st["cuda"])


def rng_digest(st: dict) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(repr(st["python"]).encode())
    h.update(np.asarray(st["numpy"][1]).tobytes())
    h.update(st["torch"].numpy().tobytes())
    if st.get("cuda") is not None:
        for t in st["cuda"]:
            h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


@dataclass
class EarlyStopper:
    min_epoch: int
    max_epoch: int
    patience: int = C.PATIENCE
    min_delta: float = C.MIN_DELTA
    anchor: float | None = None
    stale: int = 0
    best: dict | None = None      # {epoch, mae, f1, ...}
    stop_reason: str | None = None

    def update(self, epoch: int, mae: float, f1: float | None, extra: dict | None = None) -> bool:
        """Returns True when this epoch is the new best."""
        is_best = False
        if (self.best is None
                or mae < self.best["mae"] - 1e-12
                or (abs(mae - self.best["mae"]) <= 1e-12
                    and f1 is not None and self.best.get("f1") is not None
                    and f1 > self.best["f1"] + 1e-12)):
            self.best = dict(epoch=epoch, mae=float(mae),
                             f1=(None if f1 is None else float(f1)))
            if extra:
                self.best.update(extra)
            is_best = True
        if epoch < self.min_epoch:
            self.stale = 0
        elif epoch == self.min_epoch:
            self.anchor = float(self.best["mae"])
            self.stale = 0
        else:
            assert self.anchor is not None
            if mae <= self.anchor - self.min_delta:
                self.anchor = float(mae)
                self.stale = 0
            else:
                self.stale += 1
        if self.stale >= self.patience:
            self.stop_reason = f"patience{self.patience}"
        elif epoch >= self.max_epoch:
            self.stop_reason = "max_epoch"
        return is_best

    def state(self) -> dict:
        return dict(anchor=self.anchor, stale=self.stale, best=self.best,
                    stop_reason=self.stop_reason)

    @classmethod
    def from_state(cls, state: dict, min_epoch: int, max_epoch: int) -> "EarlyStopper":
        es = cls(min_epoch=min_epoch, max_epoch=max_epoch)
        es.anchor = state.get("anchor")
        es.stale = state.get("stale", 0)
        es.best = state.get("best")
        es.stop_reason = state.get("stop_reason")
        return es


class StageStore:
    """Append-only per-stage output with atomic commits."""

    LAYOUT = ["config.json", "run.json", "command.txt", "environment.json",
              "logs", "checkpoints", "commits", "val", "frozen", "cache"]

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in ("logs", "checkpoints", "commits", "val", "frozen", "cache"):
            (self.root / sub).mkdir(exist_ok=True)

    def path(self, *parts) -> Path:
        return self.root.joinpath(*parts)

    def log_event(self, obj: dict) -> None:
        obj = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), **obj)
        with open(self.root / "logs" / "train.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        print(json.dumps(obj, ensure_ascii=False, default=str)[:400], flush=True)

    def commit_epoch(self, epoch: int, payload: dict, files: dict[str, Path]) -> Path:
        """payload goes to commits/epoch-XXX.json after ckpt write + fsync."""
        ckpt_path = self.root / "checkpoints" / f"epoch-{epoch:03d}.pt"
        tmp = ckpt_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, ckpt_path)
        commit = dict(epoch=epoch, checkpoint=ckpt_path.name,
                      checkpoint_sha256=C.sha256_file(ckpt_path),
                      files={k: dict(path=str(v.relative_to(self.root)),
                                     sha256=C.sha256_file(v)) for k, v in files.items()},
                      metrics=payload.get("metrics_summary", {}),
                      early_stop=payload.get("early_stop"))
        C.write_json_atomic(self.root / "commits" / f"epoch-{epoch:03d}.json", commit)
        return ckpt_path

    def set_ref(self, name: str, epoch: int) -> None:
        C.write_json_atomic(self.root / f"{name}.json",
                            dict(epoch=epoch, checkpoint=f"checkpoints/epoch-{epoch:03d}.pt",
                                 sha256=C.sha256_file(
                                     self.root / "checkpoints" / f"epoch-{epoch:03d}.pt")))

    def latest_epoch(self) -> int:
        done = sorted(p.name for p in (self.root / "commits").glob("epoch-*.json"))
        return int(done[-1].split("-")[1].split(".")[0]) if done else 0

    def load_commit(self, epoch: int) -> dict:
        return C.load_json(self.root / "commits" / f"epoch-{epoch:03d}.json")


def run_stage(store: StageStore, recipe: dict, train_epoch_fn: Callable,
              val_fn: Callable, build_state_fn: Callable | None = None,
              resume: bool = True, budget_hours: float | None = None,
              seed: int = 42) -> dict:
    """Generic stage loop.

    train_epoch_fn(net, epoch) -> {"loss": float, ...parts}
    val_fn(net, epoch) -> (metrics_dict, rows, files_to_commit)
    build_state_fn() -> full checkpoint payload builder given (net, optimizer,
      scheduler, epoch, es, extras); None disables checkpointing (used by
      cached-head stages that manage their own persistence).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    start_epoch = 0
    es = EarlyStopper(recipe["min_epoch"], recipe["max_epoch"])
    t_start = time.time()

    if resume and store.latest_epoch() > 0:
        last = store.latest_epoch()
        commit = store.load_commit(last)
        # commit_epoch stores the bare ckpt filename; the file lives under
        # checkpoints/, so rebuild the path instead of joining the bare name
        ck = torch.load(store.root / "checkpoints" / Path(commit["checkpoint"]).name,
                        map_location="cpu", weights_only=False)
        es = EarlyStopper.from_state(ck["early_stop"], recipe["min_epoch"],
                                     recipe["max_epoch"])
        restore_rng(ck["rng"])
        build_state_fn and build_state_fn("load", ck)
        start_epoch = last
        store.log_event(dict(event="resumed", epoch=last,
                             best=es.best, stop=es.stop_reason))
        if es.stop_reason:
            return dict(status="already_complete", early_stop=es.state())

    epoch = start_epoch
    while epoch < recipe["max_epoch"]:
        epoch += 1
        if budget_hours is not None and (time.time() - t_start) / 3600 > budget_hours:
            es.stop_reason = "budget"
            store.log_event(dict(event="budget_stop", epoch=epoch - 1))
            break
        t0 = time.time()
        parts = train_epoch_fn(epoch)
        train_seconds = time.time() - t0
        t0 = time.time()
        metrics, rows, files = val_fn(epoch)
        val_seconds = time.time() - t0
        f1 = metrics.get("f1") if isinstance(metrics, dict) else None
        is_best = es.update(epoch, metrics["mae"], f1)
        rec = dict(event="epoch", epoch=epoch,
                   train_loss=parts.get("loss"),
                   **{k: v for k, v in parts.items() if k != "loss"},
                   val_mae=metrics["mae"], val_f1=f1,
                   is_best=is_best, stale=es.stale, anchor=es.anchor,
                   train_seconds=round(train_seconds, 1),
                   val_seconds=round(val_seconds, 1))
        store.log_event(rec)
        if build_state_fn is not None:
            payload = build_state_fn("save", dict(epoch=epoch, es=es, parts=parts))
            payload["metrics_summary"] = dict(mae=metrics["mae"], f1=f1)
            payload["early_stop"] = es.state()
            payload["rng"] = rng_state()
            payload["rng_digest"] = rng_digest(payload["rng"])
            store.commit_epoch(epoch, payload, files)
            if is_best:
                store.set_ref("best", epoch)
            store.set_ref("latest", epoch)
        if es.stop_reason:
            store.log_event(dict(event="stage_stop", reason=es.stop_reason,
                                 epoch=epoch, best=es.best))
            break
    return dict(status="done", early_stop=es.state(),
                seconds=time.time() - t_start)
