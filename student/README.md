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

## Who the student talks to

The student is a function: a crop goes in, a label and a confidence come out. It never
calls another part of the system, and nothing calls it except the detector.

**Detector (built, tested with a dummy model)**

| Direction | What passes | Format |
|---|---|---|
| Detector → student | one crop per item | uint8 BGR numpy array, the item's box + 10% padding, gray 128 outside the item |
| Student → detector | the classification | `{"label": "dry" \| "wet", "confidence", "dry_prob", "wet_prob", "inference_ms"}`, confidence calibrated |

The detector adds `student/` to `sys.path`, imports `classifier` and calls
`WasteClassifier(model_path=...)`, so the code stays flat files, not a package. It calls
`classify_image()` once per item (and again for each drop as it lands); the student
keeps no state between calls. Which items agree, mixed loads and drop counting are the
detector's logic, not the student's.

**Gray and plate versions.** The team's hand-off table lists the crop background as
"Proposed: gray (dataset D style); plate (dataset C) is the alternative", so there are two
student models to match two possible detector crops:

| Student | Trained on | Matches a detector that sends |
|---|---|---|
| `gray` | `dataset_D_v2_balanced` | gray-128 crops, **the only crop `item_crop()` produces today** |
| `plate` | `dataset_C_v2` | crops that keep the real plate pixels |

The `plate` crop does not exist in the detector code on `feature/multi-item-detection`
(checked on 2026-10-25; no flag or second function). If a plate version of the detector is
added, point `--model_path` at the matching student; a mismatch (a plate model on gray
crops, or the reverse) is exactly what the cross-check in `evaluate.py` measures. Either
way the interface above is the same, so switching version is only a different weights file.

**Teacher (planned, not built; the teacher is another teammate's part)**

| Direction | What passes | Status |
|---|---|---|
| Human feedback → teacher | the detector's crop, the human's label, the student's guess and confidence | Proposed. The student returns the guess and confidence but does not log or send anything. |
| Teacher → student | new weights, same MobileNetV2 design, same `classify_image()` interface | Proposed. Loading them is a file swap. |

What the teacher's weights must follow:
- **File format:** a checkpoint dict with `state_dict` and `temperature` (as `train.py`
  saves). A bare `state_dict` does not load.
- **Same preprocessing:** 224 px resize and ImageNet normalisation (`EVAL_TF` in `common.py`).
- **Own temperature:** run `calibrate.py` on the new weights before they go live.

The student never asks the teacher anything while classifying.

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

**The detector as it is today only produces gray crops.** `item_crop()` sets everything
outside the item's mask to 128, so the bin never sends a real-plate crop. The `plate`
model only matches the bin if the detector is changed to keep the plate pixels. Until
then, `gray` is the variant that matches; `plate` is kept for the comparison.

## How the detector uses the student

Checked against `feature/multi-item-detection` (`bin_vision/multi_item.py`):

| Detector side | Student side |
|---|---|
| `load_student(student_dir, model_path, "classifier")` puts the folder on `sys.path` and does `__import__("classifier")` | **Keep the flat layout** (`from common import ...`, no package, `classifier.py` defines `WasteClassifier`) |
| `WasteClassifier(model_path=...)` | one weights file; the temperature is inside it |
| `classify_image(item_crop(...))`: uint8 BGR, box + 10% padding, gray 128 outside the mask | same framing as the val/test crops here |
| reads `result["label"]` and `result["confidence"]` | `"dry"` / `"wet"` (the detector also accepts `"dry waste"` / `"wet waste"`), calibrated confidence |
| calls it once per item, and again for each drop as it lands | no state between calls; `torch.inference_mode()` is used per call and does not change global torch settings |

An empty (zero-size) crop raises a `ValueError`; any other label is rejected by the
detector with a `ValueError` of its own.

```bash
python bin_vision/multi_item.py --webcam --student_dir student --model_path student/runs/gray/weights.pt
```

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

## Training stages

1. **Base model (now).** ImageNet weights, fine-tuned on the datasets above
   (`--variant gray`, and `plate` for the comparison).
2. **Bin photos (later).** Photos captured by the bin's own camera adapt the base model
   to the real camera, lighting and plate. They are the detector's crops already
   (box + 10% padding, gray 128 outside), so they are not cropped again. Put them in the
   same layout as the datasets, then continue from the base weights:

```bash
# dataset_bin/{train,val,test}/{dry waste,wet waste}/*.jpg   (manifest.csv is optional)
python student/train.py     --variant bin --init_weights student/runs/gray/weights.pt --lr 3e-5 --out student/runs/bin
python student/calibrate.py --weights student/runs/bin/weights.pt     # re-fit T on bin photos
python student/evaluate.py  --weights student/runs/bin/weights.pt
```

`--init_weights` replaces the ImageNet start; the temperature is reset and must be
re-fitted with `calibrate.py` (it uses the bin photos' `val/`, so "0.8" means 80% on the
bin, not on the datasets). `--lr 3e-5` is a starting guess for fine-tuning, not tuned.
Tested on fake images only.

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
  that run mean nothing). `classify_image` takes about 20-55 ms per crop on a laptop CPU.
- Hand-off checked with the detector's own `load_student()`, `item_crop()` and
  `make_classify()` and a dummy (untrained) checkpoint: it loads, and gets `"dry"`/`"wet"`
  plus a confidence from a gray-128 crop. This needs a re-check with real weights.

Not done
- Real training on C v2 and D v2 balanced, and the test results.
- `evaluate.py` finds each test image's source through the manifest's `new_path`
  column (backslashes normalised, so it also works on Linux/Kaggle). The `path` column
  is the file before cleaning and does not match renamed RealWaste files.
- The gray vs plate decision.
- Who calibrates after a teacher update, and whether `WasteClassifier` should also accept
  a bare `state_dict` plus a separate temperature (it does not today).
- Logging the feedback records (crop, human label, guess, confidence) for the teacher:
  nobody owns this yet.
- Handing over the `.pt` files (`*.pt` is gitignored; they will go in a release).

## Notes

- Nothing here comes from the `ai_dustbin_2` repo.
- Data is non-commercial (RealWaste is CC BY-NC-SA 4.0); see the main README.
