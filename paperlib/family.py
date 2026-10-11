"""SQONet: P2PNet with SAF, QDPR, and OQS. Module computations match the research implementation."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import contracts as C

if str(C.P2P_SRC) not in sys.path:
    sys.path.insert(0, str(C.P2P_SRC))

CELL_IDX = torch.arange(40000) // 4  # query -> act2 cell (row-major)


def make_model_args() -> SimpleNamespace:
    return SimpleNamespace(backbone="vgg16_bn", row=2, line=2,
                           set_cost_class=C.SET_COST_CLASS,
                           set_cost_point=C.SET_COST_POINT,
                           point_loss_coef=C.POINT_LOSS_COEF,
                           eos_coef=C.EOS_COEF)


# --------------------------------------------------------------------------- #
# OQS heads
# --------------------------------------------------------------------------- #
class BQSHead(nn.Module):
    """260->32->32->4 signed residual head (conv3 zero-init)."""

    def __init__(self, input_channels: int = 260, anchors: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 1, bias=False)
        self.gn1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, 32)
        self.conv3 = nn.Conv2d(32, anchors, 1, bias=True)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.ones_(self.gn1.weight); nn.init.zeros_(self.gn1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
        nn.init.ones_(self.gn2.weight); nn.init.zeros_(self.gn2.bias)
        nn.init.zeros_(self.conv3.weight); nn.init.zeros_(self.conv3.bias)

    def forward(self, act2: torch.Tensor, base_scores: torch.Tensor) -> torch.Tensor:
        b, _, h, w = act2.shape
        score_map = base_scores.reshape(b, h, w, 4).permute(0, 3, 1, 2)
        z = self.conv3(F.gelu(self.gn2(self.conv2(
            F.gelu(self.gn1(self.conv1(torch.cat([act2, score_map], dim=1))))))))
        delta_map = 6.0 * torch.tanh(z / 6.0)
        return delta_map.permute(0, 2, 3, 1).contiguous().reshape(b, -1)


class OWLSHead(nn.Module):
    """36->16->16->4 bounded residual (out zero-init)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(36, 16, 1, bias=False)
        self.gn = nn.GroupNorm(4, 16)
        self.depthwise = nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False)
        self.out = nn.Conv2d(16, 4, 1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.ones_(self.gn.weight); nn.init.zeros_(self.gn.bias)
        nn.init.kaiming_normal_(self.depthwise.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.out.weight)

    def forward(self, hidden: torch.Tensor, margin_map: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.gn(self.conv1(torch.cat((hidden.float(), margin_map.float()), dim=1))))
        z = F.gelu(self.depthwise(z))
        residual_map = torch.tanh(self.out(z))
        return residual_map.permute(0, 2, 3, 1).contiguous().reshape(hidden.shape[0], -1)


def signed_delta(base_logits: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    half = delta / 2.0
    return torch.stack((base_logits[..., 0] - half, base_logits[..., 1] + half), dim=-1)


def bqs_parent_features(bqs_head: nn.Module, act2: torch.Tensor,
                        base_logits: torch.Tensor) -> dict:
    """BQS-R forward exposing its second hidden map (feeds OWLS)."""
    act2 = act2.float()
    base_logits = base_logits.float()
    b, _, h, w = act2.shape
    base_scores = base_logits.softmax(-1)[..., 1]
    score_map = base_scores.reshape(b, h, w, 4).permute(0, 3, 1, 2)
    first = F.gelu(bqs_head.gn1(bqs_head.conv1(torch.cat((act2, score_map), dim=1))))
    hidden = F.gelu(bqs_head.gn2(bqs_head.conv2(first)))
    delta_map = 6.0 * torch.tanh(bqs_head.conv3(hidden) / 6.0)
    delta = delta_map.permute(0, 2, 3, 1).contiguous().reshape(b, -1)
    logits = signed_delta(base_logits, delta)
    margin_map = (logits[..., 1] - logits[..., 0]).reshape(b, h, w, 4).permute(0, 3, 1, 2)
    return dict(hidden=hidden, delta=delta, logits=logits, margin_map=margin_map)


def oqs_logits(native_logits: torch.Tensor, act2: torch.Tensor,
               bqs_head: nn.Module, owls_head: nn.Module) -> torch.Tensor:
    """Final OQS 2-class logits (B,Q,2); gradients flow into the model."""
    parent = bqs_parent_features(bqs_head, act2, native_logits)
    residual = owls_head(parent["hidden"], parent["margin_map"])
    return signed_delta(parent["logits"], residual)


# --------------------------------------------------------------------------- #
# QDPR modules (arm B stack: QueryDecoder + PointwiseRefiner)
# --------------------------------------------------------------------------- #
class QueryDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj3 = nn.Conv2d(256, 64, 1)
        self.proj5 = nn.Conv2d(256, 64, 1)
        self.projA = nn.Conv2d(256, 64, 1)
        self.fc1 = nn.Linear(194, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, fpn, act2, pts, anchor):
        f3 = self.proj3(fpn[0])
        f5 = self.proj5(fpn[2])
        fa = self.projA(act2)
        gx = pts[..., 0] / 400.0 - 1.0
        gy = pts[..., 1] / 400.0 - 1.0
        grid = torch.stack([gx, gy], -1).unsqueeze(2)
        s3 = F.grid_sample(f3, grid, align_corners=False).squeeze(-1).transpose(1, 2)
        s5 = F.grid_sample(f5, grid, align_corners=False).squeeze(-1).transpose(1, 2)
        cidx = CELL_IDX.to(pts.device)
        sa = fa.flatten(2)[:, :, cidx].transpose(1, 2)
        off = (pts - anchor) / 8.0
        h = F.relu(self.fc1(torch.cat([s3, s5, sa, off], -1)))
        h = F.relu(self.fc2(h))
        delta = 6.0 * torch.tanh(self.out(h).squeeze(-1) / 6.0)
        return delta, h


class PointwiseRefiner(nn.Module):
    """Capacity-matched no-neighbour control consuming per-node features."""

    def __init__(self):
        super().__init__()
        self.pair1 = nn.Linear(135, 64)
        self.pair2 = nn.Linear(64, 64)
        self.pair_out = nn.Linear(64, 2)
        self.msg = nn.Linear(192, 64)
        self.out = nn.Linear(64, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h, pts, anchor, score):
        b, q, _ = pts.shape
        delta_out = h.new_zeros(b, q)
        for i in range(b):
            pool = pool_from_score(score[i])
            if pool.numel():
                hp = h[i, pool]
                pts_p = pts[i, pool]
                anc_p = anchor[i, pool]
                sc_p = score[i, pool]
                off = (pts_p - anc_p) / 8.0
                x = torch.cat([hp, hp * hp, sc_p[:, None], off,
                               anc_p / 400.0, pts_p / 400.0], -1)
                z = F.relu(self.pair2(F.relu(self.pair1(x))))
                gate = F.softmax(self.pair_out(z), -1)[..., 1:2]
                hnew = hp + F.relu(self.msg(torch.cat([hp, z, z * gate], -1)))
                d = 6.0 * torch.tanh(self.out(hnew).squeeze(-1) / 6.0)
                delta_out[i] = delta_out[i].index_copy(0, pool, d)
        return delta_out


POOL_SCORE = 0.12
POOL_TOP = 2048
POOL_CAP = 8192


def pool_from_score(score: torch.Tensor) -> torch.Tensor:
    """GT-free candidate pool: global top-k + score band + per-cell best."""
    order = score.argsort(descending=True)
    top = order[:min(score.numel(), POOL_TOP)]
    band = torch.nonzero(score > POOL_SCORE, as_tuple=False).squeeze(1)
    parts = [top, band]
    if score.numel() % 4 == 0:
        n_cell = score.numel() // 4
        s4 = score.view(n_cell, 4)
        parts.append(s4.argmax(1) + 4 * torch.arange(n_cell, device=score.device))
    pool = torch.unique(torch.cat(parts))
    if pool.numel() > POOL_CAP:
        keep = score[pool].argsort(descending=True)[:POOL_CAP]
        pool = pool[keep]
    return pool


# --------------------------------------------------------------------------- #
# SAF variants
# --------------------------------------------------------------------------- #
class SemanticFusion(nn.Module):
    """SAF-123 S (spatial) / G (global token mean) adapter, 58,496 params."""

    def __init__(self, arm: str):
        super().__init__()
        assert arm in ("G", "S")
        self.arm = arm
        self.local = nn.Sequential(nn.Conv2d(256, 64, 1), nn.GroupNorm(8, 64), nn.GELU())
        self.semantic = nn.Sequential(nn.Conv2d(384, 64, 1), nn.GroupNorm(8, 64), nn.GELU())
        self.mix = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, groups=64),
                                 nn.GroupNorm(8, 64), nn.GELU())
        self.out = nn.Conv2d(64, 256, 1, bias=False)
        nn.init.zeros_(self.out.weight)
        self.diagnostics: dict = {}

    def forward(self, p4: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, 384, 57, 57) fp32 (cached or online DINO path)
        semantic = F.interpolate(tokens, size=(C.SAF_GRID, C.SAF_GRID), mode="bilinear",
                                 align_corners=False)
        if self.arm == "G":
            semantic = semantic.mean((2, 3), keepdim=True).expand(-1, -1, C.SAF_GRID, C.SAF_GRID)
        residual = self.out(self.mix(self.local(p4) + self.semantic(semantic)))
        self.diagnostics = dict(
            rms=float(residual.detach().square().mean().sqrt()),
            spatial_std=float(residual.detach().flatten(2).std(2).mean()))
        return p4 + residual


class SimpleFusion(nn.Module):
    """Contract S-simple control: concat(256+384) ->1x1 640->64 (no bias)
    -> GELU -> 1x1 64->256 (no bias, zero-init); 57,344 trainable params."""

    def __init__(self):
        super().__init__()
        self.reduce = nn.Conv2d(640, 64, 1, bias=False)
        self.out = nn.Conv2d(64, 256, 1, bias=False)
        nn.init.zeros_(self.out.weight)
        self.diagnostics: dict = {}

    def forward(self, p4: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        semantic = F.interpolate(tokens, size=(C.SAF_GRID, C.SAF_GRID), mode="bilinear",
                                 align_corners=False)
        residual = self.out(F.gelu(self.reduce(torch.cat([p4, semantic], dim=1))))
        self.diagnostics = dict(
            rms=float(residual.detach().square().mean().sqrt()),
            spatial_std=float(residual.detach().flatten(2).std(2).mean()))
        return p4 + residual


def build_saf(kind):
    if kind in ("S", "G"):
        return SemanticFusion("S" if kind == "S" else "G")
    if kind == "simple":
        return SimpleFusion()
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# Family assembly
# --------------------------------------------------------------------------- #
class FamilyNet(nn.Module):
    """P2PNet + optional OQS/QDPR/SAF modules in the certified topology.

    forward(images, dino_tokens=None) ->
        dict(native_logits, logits, pts, act2, oqs=None)
      * logits: native (or QDPR-refined) 2-class logits
      * final:  logits after OQS if OQS on, else logits (native identity)
    """

    def __init__(self, oqs: bool = False, qdpr: bool = False, saf=False):
        super().__init__()
        from models import build_model
        self.use_oqs = bool(oqs)
        self.use_qdpr = bool(qdpr)
        self.saf_kind = saf if saf else None
        self.p2p = build_model(make_model_args(), training=False)
        self._act2: torch.Tensor | None = None
        self.p2p.classification.act2.register_forward_hook(self._hook_act2)
        if self.use_oqs:
            self.bqs_head = BQSHead()
            self.owls_head = OWLSHead()
        if self.use_qdpr:
            self.decoder = QueryDecoder()
            self.refiner = PointwiseRefiner()
        if self.saf_kind:
            self.saf = build_saf(self.saf_kind)

    def _hook_act2(self, _m, _i, o):
        self._act2 = o

    def forward(self, images: torch.Tensor, dino_tokens: torch.Tensor | None = None):
        feats = self.p2p.backbone(images)
        fpn = self.p2p.fpn([feats[1], feats[2], feats[3]])
        p4 = fpn[1]
        if self.saf_kind:
            if dino_tokens is None:
                raise RuntimeError("SAF arm requires DINO tokens")
            p4 = self.saf(p4, dino_tokens)
            fpn = [fpn[0], p4, fpn[2]]
        b = images.shape[0]
        regression = self.p2p.regression(p4) * 100
        classification = self.p2p.classification(p4)
        anchor = self.p2p.anchor_points(images).repeat(b, 1, 1).to(images.device)
        pts = regression + anchor
        logits = classification
        if self.use_qdpr:
            delta, h = self.decoder(fpn, self._act2, pts, anchor)
            logits = signed_delta(logits, delta)
            # pre-refiner pool score: OQS margin if OQS on, else native
            if self.use_oqs:
                inter = oqs_logits(logits, self._act2, self.bqs_head, self.owls_head)
                score1 = torch.sigmoid(inter[..., 1] - inter[..., 0]).detach()
            else:
                score1 = logits.detach().softmax(-1)[..., 1]
            delta2 = self.refiner(h, pts, anchor, score1)
            logits = signed_delta(logits, delta2)
        out = dict(native_logits=classification, logits=logits, pts=pts, act2=self._act2)
        if self.use_oqs:
            out["oqs"] = oqs_logits(logits, self._act2, self.bqs_head, self.owls_head)
        else:
            out["oqs"] = None
        return out

    # ---- inference contract: RGB only; score strictly > 0.5 --------------
    @torch.no_grad()
    def predict(self, images: torch.Tensor, dino_tokens: torch.Tensor | None = None):
        out = self.forward(images, dino_tokens)
        final = out["oqs"] if out["oqs"] is not None else out["logits"]
        scores = final.softmax(-1)[..., 1]
        return scores, out["pts"], out

    # ---- stage freeze control ---------------------------------------------
    def set_train_mode(self, stage: str):
        """stage in {p2p, bqs, owls, qdpr, saf}."""
        self.train()
        if stage in ("bqs", "owls", "qdpr", "saf"):
            self.p2p.backbone.eval()  # frozen early backbone incl. its BN
        if stage == "bqs":
            for m in (self.p2p, self.owls_head if self.use_oqs else None):
                _freeze(m)
            _unfreeze(self.bqs_head)
            self.bqs_head.train()
        elif stage == "owls":
            _freeze(self.p2p); _freeze(self.bqs_head)
            _unfreeze(self.owls_head)
            self.owls_head.train()
        elif stage == "qdpr":
            _freeze(self.p2p.backbone)
            _unfreeze(self.p2p.fpn); _unfreeze(self.p2p.classification)
            _unfreeze(self.p2p.regression)
            if self.use_oqs:
                _freeze(self.bqs_head); _freeze(self.owls_head)
            if self.use_qdpr:
                _unfreeze(self.decoder); _unfreeze(self.refiner)
        elif stage == "saf":
            _freeze(self.p2p)
            if self.use_oqs:
                _freeze(self.bqs_head); _freeze(self.owls_head)
            if self.use_qdpr:
                _freeze(self.decoder); _freeze(self.refiner)
            _unfreeze(self.saf)
            self.saf.train()
        # BN buffers stay fixed outside the p2p stage
        if stage != "p2p":
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()

    def trainable_names(self) -> list[str]:
        return [n for n, p in self.named_parameters() if p.requires_grad]


def _freeze(m):
    if m is None:
        return
    for p in m.parameters():
        p.requires_grad_(False)


def _unfreeze(m):
    if m is None:
        return
    for p in m.parameters():
        p.requires_grad_(True)


# --------------------------------------------------------------------------- #
# DINO feature path (SAF): identical binding to SAF-123 (fp32, 798, bicubic)
# --------------------------------------------------------------------------- #
_DINO = None


def dino_model():
    global _DINO
    if _DINO is None:
        import os
        os.environ["XFORMERS_DISABLED"] = "1"
        src = C.DINO_VENDOR / f"dinov2-{C.DINO_COMMIT}"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from dinov2.hub.backbones import dinov2_vits14
        model = dinov2_vits14(pretrained=False)
        weights = torch.load(C.DINO_WEIGHT, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
        model.requires_grad_(False)
        model.eval()
        _DINO = model
    return _DINO


@torch.no_grad()
def semantic_rgb(images: torch.Tensor) -> torch.Tensor:
    """(B,3,800,800) -> (B,384,57,57) fp32 via the frozen fp32 batch-one path."""
    model = dino_model()
    device = images.device
    model.to(device)
    out = []
    for image in images.split(1):
        small = F.interpolate(image, size=(C.DINO_SIZE, C.DINO_SIZE), mode="bicubic",
                              align_corners=False, antialias=True)
        tokens = model.forward_features(small)["x_norm_patchtokens"]
        assert tokens.shape == (1, C.DINO_TOKENS * C.DINO_TOKENS, C.DINO_DIM)
        feature = tokens.transpose(1, 2).reshape(1, C.DINO_DIM, C.DINO_TOKENS, C.DINO_TOKENS)
        feature = feature.half().float()   # frozen quantization binding
        assert torch.isfinite(feature).all()
        out.append(feature)
    return torch.cat(out)
