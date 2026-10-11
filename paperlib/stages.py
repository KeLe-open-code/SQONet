"""Existing stage implementations: P2PNet, BQS-R, OWLS, QDPR, and SAF. Public train.py selects the DB-only chain."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import contracts as C
from . import data as D
from . import eval as E
from . import family as fam
from .trainloop import EarlyStopper, StageStore, run_stage, rng_state, restore_rng

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def build_matcher():
    from models.matcher import HungarianMatcher_Crowd
    return HungarianMatcher_Crowd(cost_class=C.SET_COST_CLASS,
                                  cost_point=C.SET_COST_POINT)


def weighted_ce(logits2: torch.Tensor, targets: torch.Tensor, w0: float, w1: float):
    logp = F.log_softmax(logits2.float(), -1)
    nll = -logp.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)
    w = w0 + (w1 - w0) * targets.float()
    return (w * nll).sum() / w.sum().clamp(min=1e-6)


def matcher_indices(final_logits, pts, targets, matcher):
    """Official matcher with the neutral empty-GT handling."""
    nonempty = [t["point"].numel() > 0 for t in targets]
    if all(nonempty):
        return matcher({"pred_logits": final_logits, "pred_points": pts}, list(targets))
    result = []
    for i, t in enumerate(targets):
        if not nonempty[i]:
            result.append((torch.empty(0, dtype=torch.long, device=pts.device),
                           torch.empty(0, dtype=torch.long, device=pts.device)))
            continue
        sub = matcher({"pred_logits": final_logits[i:i + 1],
                       "pred_points": pts[i:i + 1]}, [t])
        result.append(sub[0])
    return result


def unified_loss(final_logits, pts, targets, matcher):
    """099-B main task loss on the FINAL logits (OQS when on, else native).

    The official matcher returns CPU int64 indices (scipy path); they are
    moved to the logit device before any gather/cat."""
    indices = matcher_indices(final_logits.detach(), pts, targets, matcher)
    dev = final_logits.device
    indices = [(src.to(dev), tgt.to(dev)) for src, tgt in indices]
    tgt = torch.zeros(final_logits.shape[:2], dtype=torch.long, device=dev)
    for i, (src, _t) in enumerate(indices):
        if src.numel():
            tgt[i, src] = 1
    loss_ce = weighted_ce(final_logits, tgt, C.EOS_COEF, 1.0)
    bidx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)]) \
        if indices else torch.zeros(0, dtype=torch.long, device=dev)
    sidx = torch.cat([src for (src, _) in indices]) if indices else \
        torch.zeros(0, dtype=torch.long, device=dev)
    src_points = pts[bidx, sidx]
    tgt_points = torch.cat([t["point"][j] for t, (_, j) in zip(targets, indices)], dim=0) \
        if any(len(j) for (_, j) in indices) else pts.new_zeros((0, 2))
    num_points = max(sum(len(t["labels"]) for t in targets), 1)
    if tgt_points.numel():
        loss_point = F.mse_loss(src_points, tgt_points, reduction="none").sum() / num_points
    else:
        loss_point = pts.sum() * 0.0
    return 1.0 * loss_ce + C.POINT_LOSS_COEF * loss_point, \
        dict(loss_ce=float(loss_ce), loss_point=float(loss_point))


def val_family(net: "fam.FamilyNet", ds: D.Dataset, split: str,
               tokens_cache: dict | None = None, store: StageStore | None = None,
               epoch: int = 0):
    """Full-split validation with rows/coords artifacts."""
    net.eval()
    rows = []
    with torch.no_grad():
        for r in ds.records[split]:
            tensor, gt, _rgb = D.load_view(r)
            tokens = None
            if net.saf_kind:
                tokens = dino_tokens_for(r.key, tensor, tokens_cache)
            scores, pts, _ = net.predict(tensor[None].to(Device), tokens)
            keep = scores[0] > 0.5
            pxy = pts[0][keep].float().cpu().numpy()
            rows.append(E.make_row(r.key, len(gt), int(keep.sum()),
                                   pred_xy=pxy, gt_xy=gt, scene=r.scene))
    metrics = E.evaluate_rows(rows)
    files: dict[str, Path] = {}
    if store is not None:
        vdir = store.root / "val" / f"epoch-{epoch:03d}"
        vdir.mkdir(parents=True, exist_ok=True)
        rowsp = vdir / "rows.jsonl"
        with open(rowsp, "w", encoding="utf-8") as f:
            for rr in E.rows_to_jsonl(rows):
                f.write(json.dumps(rr, ensure_ascii=False) + "\n")
        files["rows"] = rowsp
    return metrics, rows, files


# --------------------------------------------------------------------------- #
# DINO token cache (SAF stages)
# --------------------------------------------------------------------------- #
def dino_tokens_for(key: str, tensor: torch.Tensor, cache: dict | None):
    """(1, 384, 57, 57) tokens for a single image (cached fp16 or online)."""
    if cache is not None and key in cache:
        return cache[key].to(Device).unsqueeze(0).float()
    return fam.semantic_rgb(tensor[None].to(Device))


def build_dino_cache(store: StageStore, ds: D.Dataset, splits=("train", "val")) -> dict:
    """Cache fp16 tokens under store/cache/dino/; returns in-RAM index.

    `store` may be a shared DINO store (image-only binding, parent-free) so
    every SAF arm of every dataset reads one read-only copy; the manifest can
    therefore outlive a dataset switch, so coverage is validated on every call
    and only missing keys are extracted into the same manifest."""
    cache_dir = store.root / "cache" / "dino"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    records: dict = {}
    next_i = 0
    if manifest_path.exists():
        m = C.load_json(manifest_path)
        records = m["records"]
        used = [int(v["name"].split(".")[0]) for v in records.values()]
        next_i = max(used) + 1 if used else 0
    missing = [r for s in splits for r in ds.records[s] if r.key not in records]
    if missing:
        fam.dino_model().to(Device)
        t0 = time.time()
        for j, r in enumerate(missing):
            tensor, _gt, _rgb = D.load_view(r)
            with torch.no_grad():
                feat = fam.semantic_rgb(tensor[None].to(Device))[0].cpu().half()
            name = f"{next_i + j:04d}.npy.pt"
            torch.save(feat, cache_dir / name)
            records[r.key] = dict(name=name)
            if (j + 1) % 100 == 0:
                store.log_event(dict(event="dino_cache_progress", done=j + 1,
                                     total=len(missing),
                                     seconds=round(time.time() - t0, 1)))
        C.write_json_atomic(manifest_path, dict(
            records=records, images=len(records),
            binding="saf123-fp32-798-bicubic-antialias-fp16"))
    return {k: torch.load(cache_dir / v["name"], weights_only=True)
            for k, v in records.items()}


# --------------------------------------------------------------------------- #
# stage P: P2PNet source (QianYi) / target (DB)
# --------------------------------------------------------------------------- #
def run_p2p_stage(store: StageStore, train_ds: D.Dataset, val_ds: D.Dataset,
                  train_split: str, val_split: str, seed: int,
                  recipe_key: str, init_from: Path | None = None,
                  log_prefix: str = "") -> dict:
    recipe = dict(C.STAGE_RECIPES[recipe_key])
    net = fam.FamilyNet()
    if init_from is not None:
        obj = torch.load(init_from, map_location="cpu", weights_only=False)
        net.load_state_dict(obj["model"], strict=True)
        store.log_event(dict(event="init_from", path=str(init_from),
                             sha256=C.sha256_file(init_from)[:16]))
    net.to(Device)
    matcher = build_matcher()

    param_groups = [
        {"params": [p for n, p in net.named_parameters() if "backbone" not in n],
         "lr": recipe["lr_head"]},
        {"params": [p for n, p in net.named_parameters() if "backbone" in n],
         "lr": recipe["lr_backbone"]},
    ]
    optimizer = torch.optim.Adam(param_groups, weight_decay=recipe["weight_decay"])
    step_at = int(recipe["scheduler"].replace("step", ""))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_at, gamma=0.1)

    sched_fn = D.epoch_variant_schedule if recipe["flip"] else D.epoch_schedule_no_flip
    keys = train_ds.keys(train_split)
    by_key = {r.key: r for r in train_ds.records[train_split]}

    def train_epoch(epoch: int):
        net.train()
        sched = sched_fn(keys, seed, epoch)
        losses = []
        n_micro = 0
        for block in D.batches(sched, C.EFFECTIVE_BATCH):
            optimizer.zero_grad(set_to_none=True)
            block_losses = []
            for chunk in D.batches(block, C.MICRO_BATCH):
                tensors, gts = [], []
                for key, flip in chunk:
                    t, p, _rgb = D.load_view(by_key[key], flip=flip)
                    tensors.append(t)
                    gts.append({"point": torch.from_numpy(p).to(Device),
                                "labels": torch.ones(len(p), dtype=torch.long, device=Device)})
                batch = torch.stack(tensors).to(Device)
                out = net(batch)
                loss, parts = unified_loss(out["logits"], out["pts"], gts, matcher)
                (loss * (len(chunk) / len(block))).backward()
                block_losses.append(float(loss.detach()))
                n_micro += 1
            nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],
                                     recipe["clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.extend(block_losses)
        scheduler.step()
        return dict(loss=float(np.mean(losses)))

    def val_fn(epoch: int):
        return val_family(net, val_ds, val_split, store=store, epoch=epoch)

    def build_state(mode, payload):
        if mode == "load":
            net.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            return
        return dict(model=net.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    epoch=payload["epoch"])

    return run_stage(store, recipe, train_epoch, val_fn, build_state,
                     budget_hours=recipe["budget_hours"], seed=seed)


# --------------------------------------------------------------------------- #
# OQS cache + BQS-R + OWLS stages
# --------------------------------------------------------------------------- #
def extract_oqs_cache(store: StageStore, parent_ckpt: Path, ds: D.Dataset) -> Path:
    """Frozen-parent feature cache: train flip{0,1} + val flip0.

    Stored as a single torch file with fp16 act2 (same compromise as the
    certified BQS-004 cache; the zero-init head never consumes quantized
    act2 at step 0 because conv3 is exactly zero)."""
    out = store.root / "cache" / "oqs_features.pt"
    if out.exists():
        return out
    net = fam.FamilyNet()
    obj = torch.load(parent_ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(obj["model"], strict=True)
    net.to(Device).eval()
    records = []
    t0 = time.time()
    with torch.no_grad():
        for split in ("train", "val"):
            flips = (False, True) if split == "train" else (False,)
            for r in ds.records[split]:
                for flip in flips:
                    tensor, gt, _rgb = D.load_view(r, flip=flip)
                    logits, pts, act2 = _frozen_forward(net, tensor)
                    records.append(dict(
                        key=r.key, split=split, flip=flip, scene=r.scene,
                        logits=logits.cpu().float(),
                        points=pts.cpu().float(),
                        act2=act2.cpu().half(),
                        gt=torch.from_numpy(gt).float()))
    torch.save(records, out)
    store.log_event(dict(event="oqs_cache_built", records=len(records),
                         seconds=round(time.time() - t0, 1),
                         parent_sha=C.sha256_file(parent_ckpt)[:16]))
    return out


def _frozen_forward(net, tensor):
    with torch.no_grad():
        out = net(tensor[None].to(Device))
    return (out["logits"][0], out["pts"][0], out["act2"][0])


def load_oqs_cache(store: StageStore) -> list[dict]:
    return torch.load(store.root / "cache" / "oqs_features.pt",
                      map_location="cpu", weights_only=False)


def run_bqs_stage(store: StageStore, seed: int,
                  cache_store: StageStore | None = None) -> dict:
    """BQS-R on the cached parent features (hard Hungarian CE).

    cache_store: the oqs_cache stage store holding oqs_features.pt; when
    omitted the cache is looked up under `store` itself (smoke fixtures)."""
    recipe = dict(C.STAGE_RECIPES["bqs"])
    cached = load_oqs_cache(cache_store or store)
    train_recs = [r for r in cached if r["split"] == "train"]
    val_recs = [r for r in cached if r["split"] == "val"]
    matcher = build_matcher()
    head = fam.BQSHead().to(Device)

    # variant groups: each image has flip0/1
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(train_recs):
        groups.setdefault(r["key"], []).append(i)
    keys = sorted(groups)

    optimizer = torch.optim.AdamW(head.parameters(), lr=recipe["lr"],
                                  weight_decay=recipe["weight_decay"])

    def run_chunk(chunk, train=True):
        logits = torch.stack([r["logits"] for r in chunk]).to(Device)
        pts = torch.stack([r["points"] for r in chunk]).to(Device)
        act2 = torch.stack([r["act2"] for r in chunk]).to(Device).float()
        targets = [{"point": r["gt"].to(Device),
                    "labels": torch.ones(len(r["gt"]), dtype=torch.long, device=Device)}
                   for r in chunk]
        base_scores = logits.softmax(-1)[..., 1]
        delta = head(act2, base_scores)
        final = fam.signed_delta(logits, delta)
        if train:
            indices = matcher_indices(final, pts, targets, matcher)
            classes = torch.zeros(final.shape[:2], dtype=torch.long, device=Device)
            for bi, (src, _) in enumerate(indices):
                if src.numel():
                    classes[bi, src.to(Device)] = 1
            w = torch.tensor([C.EOS_COEF, 1.0], device=Device)
            return F.cross_entropy(final.transpose(1, 2), classes, weight=w)
        scores = final.softmax(-1)[..., 1]
        return scores

    def train_epoch(epoch: int):
        head.train()
        rng = __import__("random").Random(seed + epoch)
        chosen = [(k, rng.randrange(2)) for k in keys]
        rng.shuffle(chosen)
        order = [groups[k][v] for k, v in chosen]
        losses = []
        for block in D.batches(order, C.EFFECTIVE_BATCH):
            optimizer.zero_grad(set_to_none=True)
            block_losses = []
            for chunk in D.batches(block, C.MICRO_BATCH):
                recs = [train_recs[i] for i in chunk]
                loss = run_chunk(recs, train=True)
                (loss * (len(chunk) / len(block))).backward()
                block_losses.append(float(loss.detach()))
            nn.utils.clip_grad_norm_(head.parameters(), recipe["clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.extend(block_losses)
        return dict(loss=float(np.mean(losses)))

    def val_fn(epoch: int):
        head.eval()
        rows = []
        with torch.no_grad():
            for r in val_recs:
                scores = run_chunk([r], train=False)[0]
                keep = (scores > 0.5).cpu()
                pxy = r["points"][keep].numpy()
                rows.append(E.make_row(r["key"], len(r["gt"]), int(keep.sum()),
                                       pred_xy=pxy, gt_xy=r["gt"].numpy(),
                                       scene=r["scene"]))
        metrics = E.evaluate_rows(rows)
        files = {}
        vdir = store.root / "val" / f"epoch-{epoch:03d}"
        vdir.mkdir(parents=True, exist_ok=True)
        rowsp = vdir / "rows.jsonl"
        with open(rowsp, "w", encoding="utf-8") as f:
            for rr in E.rows_to_jsonl(rows):
                f.write(json.dumps(rr, ensure_ascii=False) + "\n")
        files["rows"] = rowsp
        return metrics, rows, files

    def build_state(mode, payload):
        if mode == "load":
            head.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            return
        return dict(model=head.state_dict(), optimizer=optimizer.state_dict(),
                    epoch=payload["epoch"])

    return run_stage(store, recipe, train_epoch, val_fn, build_state,
                     budget_hours=recipe["budget_hours"], seed=seed)


def build_owned_sets(store: StageStore, bqs_best: Path,
                     cache_store: StageStore | None = None) -> Path:
    """Frozen winner/sibling sets on every TRAIN cache variant (OWLS input).

    Writes owned_sets.pt under `store` (the owls stage store); reads
    oqs_features.pt from `cache_store` when given, else from `store`."""
    out = store.root / "cache" / "owned_sets.pt"
    if out.exists():
        return out
    cached = [r for r in load_oqs_cache(cache_store or store) if r["split"] == "train"]
    head = fam.BQSHead()
    obj = torch.load(bqs_best, map_location="cpu", weights_only=False)
    head.load_state_dict(obj["model"])
    head.to(Device).eval()
    matcher = build_matcher()
    sets_all = []
    for r in cached:
        logits = r["logits"][None].to(Device)
        with torch.no_grad():
            base_scores = logits.softmax(-1)[..., 1]
            delta = head(r["act2"][None].to(Device).float(), base_scores)
            final = fam.signed_delta(logits, delta)
        targets = [{"point": r["gt"].to(Device),
                    "labels": torch.ones(len(r["gt"]), dtype=torch.long, device=Device)}]
        indices = matcher_indices(final, r["points"][None].to(Device), targets, matcher)
        src, tgt = indices[0]
        winners = src.cpu().numpy()
        gt_pts = r["gt"].numpy()
        points = r["points"].numpy()
        prob = final.softmax(-1)[0, :, 1].cpu().numpy()
        sets = _owned_sets(points, prob, gt_pts, winners)
        sets_all.append(dict(key=r["key"], flip=r["flip"], sets=sets))
    torch.save(sets_all, out)
    store.log_event(dict(event="owned_sets_built", variants=len(sets_all),
                         bqs_sha=C.sha256_file(bqs_best)[:16]))
    return out


def _owned_sets(points, prob, gt_pts, winners, radius=10.0):
    """Port of owls_core.build_owned_sets (identical math)."""
    H = W = 100
    A = 4
    winner_set = set(map(int, winners))
    cand: dict[int, list[tuple[float, int, int]]] = {}
    for g, winner in enumerate(map(int, winners)):
        cell = int(winner) // A
        cy, cx = divmod(cell, W)
        for yy in range(max(0, cy - 1), min(H, cy + 2)):
            for xx in range(max(0, cx - 1), min(W, cx + 2)):
                for a in range(A):
                    q = A * (yy * W + xx) + a
                    if q in winner_set:
                        continue
                    d = float(np.linalg.norm(points[q].astype(np.float64)
                                             - gt_pts[g].astype(np.float64)))
                    if d <= radius:
                        cand.setdefault(q, []).append((0.05 * d - float(prob[q]), g, q))
    siblings: dict[int, list[int]] = {}
    for q, choices in cand.items():
        _, g, _ = min(choices)
        siblings.setdefault(int(g), []).append(int(q))
    result = []
    seen = set()
    for g, winner in enumerate(map(int, winners)):
        queries = [winner, *sorted(siblings.get(g, []))]
        overlap = seen.intersection(queries)
        if overlap:
            raise RuntimeError(f"owned-set overlap {sorted(overlap)[:3]}")
        seen.update(queries)
        result.append(dict(gt_index=int(g), winner=int(winner), queries=queries,
                           eligible=len(queries) > 1))
    return result


def run_owls_stage(store: StageStore, seed: int,
                   cache_store: StageStore | None = None,
                   bqs_store: StageStore | None = None) -> dict:
    """OWLS on owned sets (winner log-softmax, set mean then image mean).

    cache_store: oqs_cache stage store for oqs_features.pt; bqs_store: bqs
    stage store whose best.json locates the frozen BQS head.  Both default
    to `store` for smoke fixtures."""
    recipe = dict(C.STAGE_RECIPES["owls"])
    cached = load_oqs_cache(cache_store or store)
    train_recs = [r for r in cached if r["split"] == "train"]
    val_recs = [r for r in cached if r["split"] == "val"]
    sets_by_key = {}
    for entry in torch.load(store.root / "cache" / "owned_sets.pt",
                            map_location="cpu", weights_only=False):
        sets_by_key[(entry["key"], entry["flip"])] = entry["sets"]
    bqs = fam.BQSHead()
    bqs_obj = torch.load(bqs_best_path(bqs_store or store), map_location="cpu",
                         weights_only=False)
    bqs.load_state_dict(bqs_obj["model"])
    bqs.to(Device).eval()
    for p in bqs.parameters():
        p.requires_grad_(False)

    owls = fam.OWLSHead().to(Device)
    optimizer = torch.optim.AdamW(owls.parameters(), lr=recipe["lr"],
                                  weight_decay=recipe["weight_decay"])
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(train_recs):
        groups.setdefault(r["key"], []).append(i)
    keys = sorted(groups)

    def margins_for(recs):
        logits = torch.stack([r["logits"] for r in recs]).to(Device)
        act2 = torch.stack([r["act2"] for r in recs]).to(Device).float()
        with torch.no_grad():
            parent = fam.bqs_parent_features(bqs, act2, logits)
        residual = owls(parent["hidden"], parent["margin_map"])
        final = fam.signed_delta(parent["logits"], residual)
        return final, (final[..., 1] - final[..., 0])

    def train_epoch(epoch: int):
        owls.train()
        rng = __import__("random").Random(seed + epoch)
        chosen = [(k, rng.randrange(2)) for k in keys]
        rng.shuffle(chosen)
        order = [groups[k][v] for k, v in chosen]
        losses = []
        for block in D.batches(order, C.EFFECTIVE_BATCH):
            optimizer.zero_grad(set_to_none=True)
            block_losses = []
            for chunk in D.batches(block, C.MICRO_BATCH):
                recs = [train_recs[i] for i in chunk]
                _final, margins = margins_for(recs)
                img_losses = []
                for bi, r in enumerate(recs):
                    sets = sets_by_key[(r["key"], r["flip"])]
                    per_set = []
                    for s in sets:
                        qs = s["queries"]
                        if len(qs) <= 1:
                            continue
                        idx = torch.as_tensor(qs, dtype=torch.long, device=Device)
                        wpos = qs.index(s["winner"])
                        per_set.append(-F.log_softmax(margins[bi].index_select(0, idx), 0)[wpos])
                    img_losses.append(torch.stack(per_set).mean() if per_set
                                      else margins[bi].sum() * 0.0)
                loss = torch.stack(img_losses).mean()
                (loss * (len(chunk) / len(block))).backward()
                block_losses.append(float(loss.detach()))
            nn.utils.clip_grad_norm_(owls.parameters(), recipe["clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.extend(block_losses)
        return dict(loss=float(np.mean(losses)))

    def val_fn(epoch: int):
        owls.eval()
        rows = []
        with torch.no_grad():
            for r in val_recs:
                final, _m = margins_for([r])
                scores = final.softmax(-1)[0, :, 1]
                keep = (scores > 0.5).cpu()
                pxy = r["points"][keep].numpy()
                rows.append(E.make_row(r["key"], len(r["gt"]), int(keep.sum()),
                                       pred_xy=pxy, gt_xy=r["gt"].numpy(),
                                       scene=r["scene"]))
        metrics = E.evaluate_rows(rows)
        files = {}
        vdir = store.root / "val" / f"epoch-{epoch:03d}"
        vdir.mkdir(parents=True, exist_ok=True)
        rowsp = vdir / "rows.jsonl"
        with open(rowsp, "w", encoding="utf-8") as f:
            for rr in E.rows_to_jsonl(rows):
                f.write(json.dumps(rr, ensure_ascii=False) + "\n")
        files["rows"] = rowsp
        return metrics, rows, files

    def build_state(mode, payload):
        if mode == "load":
            owls.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            return
        return dict(model=owls.state_dict(), optimizer=optimizer.state_dict(),
                    epoch=payload["epoch"])

    return run_stage(store, recipe, train_epoch, val_fn, build_state,
                     budget_hours=recipe["budget_hours"], seed=seed)


def bqs_best_path(store: StageStore) -> Path:
    best = C.load_json(store.root / "best.json")
    return store.root / best["checkpoint"]


# --------------------------------------------------------------------------- #
# QDPR (ON) / Q-OFF continuation, SAF stages — online forward
# --------------------------------------------------------------------------- #
def assemble_chain_ckpt(base_ckpt: Path, bqs_ckpt: Path, owls_ckpt: Path,
                        out_path: Path) -> Path:
    """Merge the shared chain pieces into one full-FamilyNet checkpoint.

    The chain stores each trained head separately: the p2p stage saves the
    full net (p2p.* keys) while the bqs/owls stages save only their head
    params under bare module names.  Downstream arms need a single parent
    carrying p2p.* + bqs_head.* + owls_head.* for _assemble_full."""
    if out_path.exists():
        return out_path
    state = dict(torch.load(base_ckpt, map_location="cpu",
                            weights_only=False)["model"])
    for ckpt, prefix in ((bqs_ckpt, "bqs_head."), (owls_ckpt, "owls_head.")):
        head = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        for k, v in head.items():
            state[prefix + k] = v
    net = fam.FamilyNet(oqs=True, qdpr=False)
    missing, unexpected = net.load_state_dict(state, strict=False)
    allowed = {n for n in missing
               if n.startswith(("decoder.", "refiner."))}
    assert not unexpected and set(missing) == allowed, (missing, unexpected)
    torch.save(dict(model=state), out_path)
    return out_path


def _assemble_full(oqs_on: bool, qdpr_on: bool, saf_kind, parent_ckpt: Path,
                   strict: bool = True):
    net = fam.FamilyNet(oqs=oqs_on, qdpr=qdpr_on, saf=saf_kind)
    obj = torch.load(parent_ckpt, map_location="cpu", weights_only=False)
    state = obj["model"]
    if strict:
        net.load_state_dict(state, strict=True)
    else:
        missing, unexpected = net.load_state_dict(state, strict=False)
        # only NEW modules of this stage may be absent; an OQS/QDPR parent
        # must carry its own heads
        allowed = {n for n in missing if n.startswith(("decoder.", "refiner.", "saf."))}
        assert not unexpected and set(missing) == allowed, (missing, unexpected)
    return net, obj


def run_qdpr_stage(store: StageStore, train_ds: D.Dataset, val_ds: D.Dataset,
                   seed: int, oqs_on: bool, qdpr_on: bool,
                   parent_ckpt: Path) -> dict:
    """Unified Q stage: qdpr_on=True instantiates decoder+refiner (099-B);
    False keeps the parent training FPN/cls/reg with the SAME recipe so the
    arms differ only by the module (contract section 4)."""
    recipe = dict(C.STAGE_RECIPES["qdpr"])
    net, _obj = _assemble_full(oqs_on, qdpr_on, False, parent_ckpt,
                               strict=False)
    net.to(Device)
    net.set_train_mode("qdpr")
    matcher = build_matcher()

    old_params = [p for n, p in net.named_parameters()
                  if n.startswith("p2p.") and p.requires_grad]
    new_params = [p for n, p in net.named_parameters()
                  if not n.startswith("p2p.") and p.requires_grad]
    groups = [dict(params=old_params, lr=recipe["lr_old"]),
              dict(params=new_params, lr=recipe["lr_new"])] \
        if (qdpr_on and new_params) else [dict(params=old_params, lr=recipe["lr_old"])]
    optimizer = torch.optim.AdamW(groups, weight_decay=recipe["weight_decay"])

    keys = train_ds.keys("train")
    by_key = {r.key: r for r in train_ds.records["train"]}

    def train_epoch(epoch: int):
        net.set_train_mode("qdpr")
        sched = D.epoch_schedule_no_flip(keys, seed, epoch)
        losses = []
        for block in D.batches(sched, C.EFFECTIVE_BATCH):
            optimizer.zero_grad(set_to_none=True)
            block_losses = []
            for chunk in D.batches(block, C.MICRO_BATCH):
                tensors, gts = [], []
                for key, _flip in chunk:
                    t, p, _rgb = D.load_view(by_key[key])
                    tensors.append(t)
                    gts.append({"point": torch.from_numpy(p).to(Device),
                                "labels": torch.ones(len(p), dtype=torch.long, device=Device)})
                batch = torch.stack(tensors).to(Device)
                out = net(batch)
                final = out["oqs"] if out["oqs"] is not None else out["logits"]
                loss, parts = unified_loss(final, out["pts"], gts, matcher)
                (loss * (len(chunk) / len(block))).backward()
                block_losses.append(float(loss.detach()))
            nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],
                                     recipe["clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.extend(block_losses)
        return dict(loss=float(np.mean(losses)))

    def val_fn(epoch: int):
        return val_family(net, val_ds, "val", store=store, epoch=epoch)

    def build_state(mode, payload):
        if mode == "load":
            net.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            return
        return dict(model=net.state_dict(), optimizer=optimizer.state_dict(),
                    epoch=payload["epoch"],
                    parent_sha=C.sha256_file(parent_ckpt)[:16])

    return run_stage(store, recipe, train_epoch, val_fn, build_state,
                     budget_hours=recipe["budget_hours"], seed=seed)


def run_saf_stage(store: StageStore, train_ds: D.Dataset, val_ds: D.Dataset,
                  seed: int, oqs_on: bool, qdpr_on: bool, saf_kind,
                  parent_ckpt: Path, dino_store: StageStore | None = None) -> dict:
    """SAF (or its controls): train ONLY the fusion adapter; DINO and the
    whole parent chain are frozen; zero-init residual replays the parent at
    step 0."""
    recipe = dict(C.STAGE_RECIPES["saf"])
    net, _obj = _assemble_full(oqs_on, qdpr_on, saf_kind, parent_ckpt, strict=False)
    net.to(Device)
    net.set_train_mode("saf")
    matcher = build_matcher()
    optimizer = torch.optim.AdamW(net.saf.parameters(), lr=recipe["lr"],
                                  weight_decay=recipe["weight_decay"])

    tokens_cache = build_dino_cache(dino_store or store, train_ds)
    keys = train_ds.keys("train")
    by_key = {r.key: r for r in train_ds.records["train"]}

    def train_epoch(epoch: int):
        net.set_train_mode("saf")
        sched = D.epoch_schedule_no_flip(keys, seed, epoch)
        losses = []
        for block in D.batches(sched, C.EFFECTIVE_BATCH):
            optimizer.zero_grad(set_to_none=True)
            block_losses = []
            for chunk in D.batches(block, C.MICRO_BATCH):
                tensors, gts = [], []
                for key, _flip in chunk:
                    t, p, _rgb = D.load_view(by_key[key])
                    tensors.append(t)
                    gts.append({"point": torch.from_numpy(p).to(Device),
                                "labels": torch.ones(len(p), dtype=torch.long, device=Device)})
                batch = torch.stack(tensors).to(Device)
                tokens = torch.stack([tokens_cache[k] for k, _ in chunk]).to(Device).float()
                out = net(batch, tokens)
                final = out["oqs"] if out["oqs"] is not None else out["logits"]
                loss, parts = unified_loss(final, out["pts"], gts, matcher)
                (loss * (len(chunk) / len(block))).backward()
                block_losses.append(float(loss.detach()))
            nn.utils.clip_grad_norm_(list(net.saf.parameters()), recipe["clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.extend(block_losses)
        return dict(loss=float(np.mean(losses)),
                    saf_rms=float(np.mean([net.saf.diagnostics.get("rms", 0.0)])))

    def val_fn(epoch: int):
        return val_family(net, val_ds, "val", tokens_cache=tokens_cache,
                          store=store, epoch=epoch)

    def build_state(mode, payload):
        if mode == "load":
            net.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            return
        return dict(model=net.state_dict(), optimizer=optimizer.state_dict(),
                    epoch=payload["epoch"],
                    parent_sha=C.sha256_file(parent_ckpt)[:16])

    return run_stage(store, recipe, train_epoch, val_fn, build_state,
                     budget_hours=recipe["budget_hours"], seed=seed)
