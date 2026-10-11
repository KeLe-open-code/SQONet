# Reproduction notes

## Modules and ablations

| Arm | SAF | QDPR | OQS |
|---|:---:|:---:|:---:|
| A0 / P2PNet | | | |
| A1 | | | ✓ |
| A2 | | ✓ | |
| A3 | ✓ | | |
| A4 | | ✓ | ✓ |
| A5 | ✓ | | ✓ |
| A6 | ✓ | ✓ | |
| A7 / SQONet | ✓ | ✓ | ✓ |

The full DB chain trains P2PNet, BQS-R, OWLS, QDPR, and SAF in that order. The reporting order of the three contributions is SAF, QDPR, OQS. The public launcher uses DB train/validation data directly, with ImageNet backbone initialization and frozen self-supervised DINOv2 weights; it does not introduce another supervised tree dataset as a source stage.

`paperlib/family.py` retains the research model computations, including bounded score residuals, candidate pooling, frozen semantic tokens, and the strict `score > 0.5` rule. QDPR refines query scores; its name must not be interpreted as adding an independent coordinate-regression head. Coordinates come from the P2PNet regression branch.

## Training configuration

Training constants and stage-specific optimizer settings are in `paperlib/contracts.py`. The DB QDPR stage uses the original horizontal-flip schedule in `paperlib/qdpr_train.py`; SAF uses the unflipped schedule. The effective batch is eight, accumulated from micro-batches of two. All stages use FP32. The baseline point-loss coefficient is an explicit CLI parameter, defaulting to `0.0001`; the later QDPR/SAF stages use `0.0002`. RNGs are set before constructing each stage's model.

Validation MAE selects checkpoints, with localization F1 as a tie-breaker. Saved `best.json` files bind the selected epoch to its SHA-256 hash. The stage loop retains its original early-stopping behavior and writes resumable training checkpoints. Cache and checkpoint storage can be substantial; choose an output drive with adequate free space.

The public launcher adapts local paths and exposes the existing stage functions. Its syntax, imports, model/checkpoint compatibility, and synthetic inference/loss paths have been checked. The publication release does not represent an additional complete training run; numerical reproduction on other hardware and library versions is not guaranteed to be bitwise identical.

## Checkpoints and paper summaries

This is a code-only release: model weights and datasets are not included. For inference, provide your own checkpoint with tensor model state under the `model` key. `tools/export_checkpoint.py` converts a trusted local training checkpoint into that format. The frozen DINOv2 dependency is separate.

The manuscript's four-run DB ablation summary includes the historical progressive run and the later runs with seeds 42, 2026, and 3407. That historical chain and the later three-seed runs have different initialization histories. Running this public launcher with three seeds alone does not recreate the four-run historical summary.

## Checks without the DB images

```bash
python tests/check_release.py
```

This checks Hungarian matching and counting-metric edge cases on synthetic points. It does not need any dataset or model weights and does not produce new experimental scores.

## Provenance

`provenance/source_files.json` records source/release hashes for copied components. Release differences consist of portable paths, a no-download checkpoint-loading option for the VGG backbone, and public entry points. Model module calculations, original preprocessing primitives, and assignment/metric code are retained.
