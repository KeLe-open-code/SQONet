# SQONet

Research code for **SQONet: Semantic Fusion and Query Scoring for Tree Counting and Localization in Urban and Suburban Remote Sensing Images**.

Le Ke, Yang Liu, Xixin Chen, Guoxiong Zhou, Genhua Liu, Huaiqing Zhang, Jiang Zhang, Mingfang Wang, and Liujun Li.

SQONet integrates three modules into a direct point-prediction framework:

- **SAF**: semantic-augmented fusion of frozen DINOv2 spatial features.
- **QDPR**: query decoding and pointwise refinement of retention scores.
- **OQS**: ownership-aware query scoring using bounded residual calibration and locally exclusive supervision.

The network predicts tree locations and obtains the count from the retained point set. Inference uses only RGB imagery. The release retains the original module computations, loss functions, preprocessing, and Hungarian-based evaluator.

## Release contents

- Model, training stages, prediction and evaluation entry points.
- A source-code archive in [Releases](https://github.com/KeLe-open-code/SQONet/releases).
- Input-format documentation for privately held images, annotations, split lists, and checkpoints.
- Source checksums under `provenance/`.

**This repository distributes code only. It does not include model weights, images, annotations, or dataset split files.**

## Installation

Use Python 3.10 or later and install a PyTorch/torchvision pair suitable for your hardware, followed by:

```bash
pip install -r requirements.txt
```

The release was checked locally with Python 3.13, PyTorch 2.11.0, torchvision 0.26.0, OpenCV 5.0.0, and SciPy 1.18.0. CPU synthetic checks cover the model and matching implementation; they are not a new training run or a re-evaluation of the test set. See [reproduction notes](docs/REPRODUCTION.md).

## Model weights

Provide your own trained SQONet checkpoint as `weights/sqonet.pt`. Inference expects a tensor state dictionary under the `model` key. For trusted local training checkpoints containing optimizer/RNG metadata, use `tools/export_checkpoint.py` to produce this inference format.

The frozen semantic encoder is the original DINOv2 **ViT-S/14** model. Its weights can be obtained separately from the [official DINOv2 repository](https://github.com/facebookresearch/dinov2#pretrained-models) and saved as `weights/dinov2_vits14_pretrain.pth`.

```bash
mkdir -p weights
curl -L https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth -o weights/dinov2_vits14_pretrain.pth
```

The required ViT-S/14 inference source files from DINOv2 commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8` are included under `third_party/`; no source code is downloaded at inference time. Neither SQONet nor DINOv2 weights are uploaded to this repository or its releases.

## Predict an image

```bash
python predict.py --image /path/to/image.jpg --checkpoint weights/sqonet.pt --output outputs/example --device cpu
```

Use `--device cuda` on a compatible GPU. The outputs are `points.json` and `prediction.png`. JSON records both the 800 × 800 inference coordinates and coordinates mapped to the input image. No annotations are read by this command. The score threshold is strictly greater than 0.5, with no extra non-maximum suppression.

## Train and evaluate

Prepare your authorized local dataset using the [input-format instructions](docs/DATA_FORMAT.md). Images, annotations, and split lists must be supplied locally. The launcher does not download a dataset or create a new split.

```bash
python train.py --data-root data/DB --output outputs/db_seed42 --seed 42 --arm A7 --device cuda
python evaluate.py --data-root data/DB --checkpoint weights/sqonet.pt --split val --output outputs/val_seed42 --device cuda
```

Repeat training with `--seed 2026` and `--seed 3407` in separate output directories. `--arm A0` runs the P2PNet baseline; A1–A7 select the module combinations in [reproduction notes](docs/REPRODUCTION.md). Training reads **train/val only**. Evaluate `--split test` only after the model and all choices have been fixed. Existing output directories are protected unless training is explicitly resumed.

## Licensing and attribution

This is a **public academic-research code release**. P2PNet's upstream license restricts use to academic research; this repository does not remove that restriction or grant commercial rights. DINOv2 retains its Apache-2.0 license. No dataset or model-weight license is granted by this code-only release. Read [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Please cite the SQONet manuscript when using this work. Publication metadata will be updated when available; the repository does not claim acceptance or an assigned DOI.

## Acknowledgments

This work was supported in part by the Hunan Provincial Natural Science Foundation of China under Grant 2025JJ50385 and in part by the National Natural Science Foundation of China under Grant 62276276.
