# Local data format

No dataset files, split lists, annotations, or image pixels are distributed with this code. Supply your own authorized local files. The loader expects this layout:

```text
data/DB/
  images/
    DongBei_city/<image>.jpg
    DongBei_suburbs/<image>.jpg
  annotations/
    DongBei_city/<image>.json
    DongBei_suburbs/<image>.json
  splits/
    train.txt
    val.txt
    test.txt
```

Each split-list line contains an image path and an annotation path, separated by whitespace and relative to `images/` and `annotations/`, respectively. For example, using a fictitious filename:

```text
DongBei_city/example.jpg DongBei_city/example.json
```

Annotations may use LabelMe point shapes or the simplified structure `{"points": [[x, y], ...]}`. Coordinates are in original-image pixels. Preserve the dataset's established train/validation/test split. The DB paper uses 775/166/167 valid images; this release neither supplies nor regenerates those lists.

Images are resized to 800 × 800 using OpenCV linear interpolation. Point coordinates scale to the same canvas. The final point set contains predictions whose score is strictly greater than 0.5. Localization evaluation first solves the full minimum-total-distance Hungarian assignment, then counts assigned distances at or below the selected tolerance. F1@10 uses 10 pixels in the resized canvas, and aggregates TP, prediction counts, and GT counts over the split.

When using other authorized data, provide the same structure and document its own split, dimensions, and imagery rights. A format-compatible dataset is not automatically the DB benchmark and should not be reported as such.
