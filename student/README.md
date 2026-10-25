# Student model

The student is the small, fast wet/dry classifier that runs on the bin. This folder
holds its training, calibration, evaluation and inference code. It is built on the
`student` branch.

**Status: code written and smoke-tested on fake images only. Not yet trained on the real
datasets, so there are no results yet.**

## Scope

The student does one thing: it takes the picture of **one** item and says wet or dry.

- Input: one cropped item (BGR numpy image). The detector does the cropping.
- Output: a label and a calibrated confidence.
- Not in this folder: cropping, item counting, the "mixed load" logic and drop counting.
  Those belong to the detector.

## Interface

```python
from classifier import WasteClassifier

classifier = WasteClassifier(model_path="student/runs/gray/weights.pt")
result = classifier.classify_image(crop_bgr)    # numpy BGR image of ONE item
# {"label": "dry" | "wet", "confidence": 0.0-1.0,
#  "dry_prob": ..., "wet_prob": ..., "inference_ms": ...}
```

The detector (`bin_vision/multi_item.py`) loads this with
`--student_dir student --model_path student/runs/gray/weights.pt` (module `classifier`).

`confidence` is the probability of the predicted label after temperature scaling, so
"0.8" should mean about 80% likely to be right. The human-in-the-loop threshold depends
on this. The temperature is stored inside the weights file, so one file is enough.

## Model

MobileNetV2, starting from ImageNet weights, with a 2-class head (`dry`, `wet`; the
dataset folders are `dry waste` and `wet waste`).
All layers are fine-tuned (AdamW, lr 1e-4, cosine schedule, 15 epochs, class-weighted
loss, seed 42). The epoch with the best validation accuracy is kept.

## Two versions: gray and plate

The crop the detector hands over has either a gray or a real-plate background. The
student must be trained the way it will be used, and the team has not yet decided which,
so both are prepared. They use the same images, seed and settings; only the background
differs.

| Variant | Dataset | Background around the item |
|---|---|---|
| `gray` | `dataset_D_v2_balanced` | background removed, flat gray **128** |
| `plate` | `dataset_C_v2` | the real plate |

If `gray` is chosen, the detector's `item_crop()` must fill the background with the same
value (128) that `remove_background.py` used. It does.

## Item crops (same framing as the bin)

On the bin the student never sees a whole photo: the detector sends the item's
bounding box + 10% padding. The dataset images are whole photos where the item often
fills only a quarter of the frame. So every image is first cropped to its item, using
the masks in `dataset_D_v2_balanced/masks/` (they also fit `dataset_C_v2`: same file
names):

| Split | Padding around the item's box |
|---|---|
| train | random 5-30% of the box's longer side |
| val, test | exactly 10%, as the detector |

The crop is clamped to the image, like the detector's. Masks that cover under 1% of
the image are failed cut-outs; those images are used whole. `train.py` stores the mask
folder in the weights file, so `calibrate.py` and `evaluate.py` crop the same way.
`--no_crop` trains on whole images, for comparison.

## Files

| File | What it does |
|---|---|
| `common.py` | Constants (gray 128, 224 px, ImageNet normalisation), model builder, dataset loader |
| `train.py` | Fine-tunes the model. Uses `train/` to fit and `val/` to pick the epoch. Never reads `test/`. |
| `calibrate.py` | Temperature scaling on `val/`. Writes T into the weights file and `temperature.json`. Prints NLL and ECE before and after. |
| `classifier.py` | `WasteClassifier` (the interface above) |
| `evaluate.py` | Test results: accuracy overall and per source, confusion matrix, ECE. Can be pointed at the other variant's test set as a cross-check. |

## How to run

```bash
pip install -r requirements.txt

# datasets: see the main README (release data-v2), unzipped into the repo folder.
# dataset_D_v2_balanced is needed for both variants (it holds the masks).
python student/train.py     --variant gray  --data_root .
python student/calibrate.py --weights student/runs/gray/weights.pt
python student/evaluate.py  --weights student/runs/gray/weights.pt

# same for --variant plate, then cross-check each model on the other's test set:
python student/evaluate.py  --weights student/runs/gray/weights.pt --data_dir dataset_C_v2
```

`test/` is only for final numbers. Run `evaluate.py` once the settings are fixed.

## Updates from the teacher

The teacher will later send new weights with the same model design and the same
`classify_image()` interface; loading them is just a file swap. **Each new weights file
needs its own temperature**: run `calibrate.py` on it before it goes live. An old
temperature on new weights gives wrong confidences and breaks the human-in-the-loop
threshold. (Who runs this, the teacher or the student side, is still to be agreed.)

## Done and not done

Done
- Training, calibration, evaluation and inference code.
- Smoke test of the whole chain on a small set of random fake images (the numbers from
  that run mean nothing). `classify_image` takes about 54 ms per crop on a laptop CPU.

Not done
- Real training on C v2 and D v2 balanced, and the test results.
- `evaluate.py` finds each test image's source through the manifest's `new_path`
  column (backslashes normalised, so it also works on Linux/Kaggle). The `path` column
  is the file before cleaning and does not match renamed RealWaste files.
- The gray vs plate decision.
- Who calibrates after a teacher update.
- Handing over the `.pt` files (`*.pt` is gitignored; they will go in a release).

## Notes

- Nothing here comes from the `ai_dustbin_2` repo.
- Data is non-commercial (RealWaste is CC BY-NC-SA 4.0); see the main README.
