"""DB horizontal-flip training stage, extracted without computational changes."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from . import contracts as C, data as D, stages as S, trainloop as TL
Device = S.Device

def run_qdpr_stage_flip(store: TL.StageStore, train_ds, val_ds, seed: int,
                        oqs_on: bool, qdpr_on: bool, parent_ckpt: Path) -> dict:
    """paperlib stages.run_qdpr_stage verbatim except the schedule: the
    original horizontal-flip schedule (recipe_registry db_qdpr /
    internal_qdpr augmentation), matching rqd2_099 epoch_variant_schedule."""
    recipe = dict(C.STAGE_RECIPES["qdpr"])
    net, _obj = S._assemble_full(oqs_on, qdpr_on, False, parent_ckpt, strict=False)
    net.to(Device)
    net.set_train_mode("qdpr")
    matcher = S.build_matcher()

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
        sched = D.epoch_variant_schedule(keys, seed, epoch)   # <- the flip change
        losses = []
        for block in D.batches(sched, C.EFFECTIVE_BATCH):
            optimizer.zero_grad(set_to_none=True)
            for chunk in D.batches(block, C.MICRO_BATCH):
                tensors, gts = [], []
                for key, flip in chunk:
                    t, p, _rgb = D.load_view(by_key[key], flip=flip)
                    tensors.append(t)
                    gts.append({"point": torch.from_numpy(p).to(Device),
                                "labels": torch.ones(len(p), dtype=torch.long,
                                                     device=Device)})
                batch = torch.stack(tensors).to(Device)
                out = net(batch)
                final = out["oqs"] if out["oqs"] is not None else out["logits"]
                loss, parts = S.unified_loss(final, out["pts"], gts, matcher)
                (loss * (len(chunk) / len(block))).backward()
                losses.append(float(loss.detach()))
            nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],
                                     recipe["clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return dict(loss=float(np.mean(losses)))

    def val_fn(epoch: int):
        return S.val_family(net, val_ds, "val", store=store, epoch=epoch)

    def build_state(mode, payload):
        if mode == "load":
            net.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            return
        return dict(model=net.state_dict(), optimizer=optimizer.state_dict(),
                    epoch=payload["epoch"],
                    parent_sha=C.sha256_file(parent_ckpt)[:16])

    return TL.run_stage(store, recipe, train_epoch, val_fn, build_state,
                        budget_hours=recipe["budget_hours"], seed=seed)
