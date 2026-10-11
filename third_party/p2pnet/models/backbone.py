# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import os
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

import models.vgg_ as models

class BackboneBase_VGG(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if name == 'vgg16_bn':
                self.body1 = nn.Sequential(*features[:13])
                self.body2 = nn.Sequential(*features[13:23])
                self.body3 = nn.Sequential(*features[23:33])
                self.body4 = nn.Sequential(*features[33:43])
            else:
                self.body1 = nn.Sequential(*features[:9])
                self.body2 = nn.Sequential(*features[9:16])
                self.body3 = nn.Sequential(*features[16:23])
                self.body4 = nn.Sequential(*features[23:30])
        else:
            if name == 'vgg16_bn':
                self.body = nn.Sequential(*features[:44])  # 16x down-sample
            elif name == 'vgg16':
                self.body = nn.Sequential(*features[:30])  # 16x down-sample
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers

    def forward(self, tensor_list):
        # Handle NestedTensor: extract underlying tensor
        if hasattr(tensor_list, 'tensors'):
            xs = tensor_list.tensors
        else:
            xs = tensor_list

        out = []
        if self.return_interm_layers:
            for _, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
                xs = layer(xs)
                out.append(xs)
        else:
            xs = self.body(xs)
            out.append(xs)
        return out


_PD = Path(__file__).resolve().parent.parent / "pretrained_weights"
_LW = {"vgg16_bn": "vgg16_bn-6c64b313.pth", "vgg16": "vgg16-397923af.pth"}

class Backbone_VGG(BackboneBase_VGG):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, return_interm_layers: bool):
        # Prefer bundled pretrained weights (offline, single-folder upload);
        # fall back to torchvision download.
        import torchvision.models as tv_models
        lp = _PD / _LW.get(name, "")
        if os.environ.get("SQONET_PRETRAINED", "1") == "0":
            ctor = tv_models.vgg16_bn if name == "vgg16_bn" else tv_models.vgg16
            backbone = ctor(weights=None)
        elif name in _LW and lp.exists():
            ctor = tv_models.vgg16_bn if name == "vgg16_bn" else tv_models.vgg16
            backbone = ctor(weights=None)
            backbone.load_state_dict(torch.load(lp, map_location="cpu"))
        elif name == "vgg16_bn":
            backbone = tv_models.vgg16_bn(weights="DEFAULT")
        elif name == "vgg16":
            backbone = tv_models.vgg16(weights="DEFAULT")
        else:
            raise ValueError(f"Unknown backbone: {name}")
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers)


def build_backbone(args):
    backbone = Backbone_VGG(args.backbone, True)
    return backbone

if __name__ == '__main__':
    Backbone_VGG('vgg16', True)