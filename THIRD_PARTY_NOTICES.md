# Third-party notices

## P2PNet and DETR-derived utilities

- Upstream: https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet
- Copyright (C) 2021 THL A29 Limited, a Tencent company.
- The P2PNet terms expressly restrict use to academic research. The complete upstream license, including its Apache-2.0 third-party notices, is retained at `third_party/p2pnet/LICENSE`.
- The vendored tree includes the project's existing tensor/NestedTensor and torchvision compatibility adaptations. Public-release changes add an environment-controlled option to construct the VGG backbone without downloading ImageNet weights when a complete checkpoint will immediately be loaded. Network layers are unchanged.
- Original Facebook/DETR copyright notices remain in the applicable files.

## DINOv2

- Upstream: https://github.com/facebookresearch/dinov2
- Frozen source commit: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- Only the ViT-S/14 inference import dependencies are included, unchanged. Unrelated Cell-DINO, XRay-DINO model files, evaluation tools, and training applications are excluded.
- Copyright Meta Platforms, Inc. and affiliates; Apache License 2.0. See the retained license in `third_party/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8/`.
- Frozen ViT-S/14 weights are obtained separately from the official upstream download. The dependency is not a tree-dataset supervised pretraining stage.

## DB source imagery

DB image pixels originate from third-party satellite imagery. No dataset files (including images, point annotations, or fixed split lists) are distributed here. The code license does not convey rights to Google or its imagery providers' content.
