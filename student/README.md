# Student model

The student is the small, fast wet/dry classifier that runs on the bin. This folder holds
its training, calibration, evaluation, inference and update code. It is built on the
`student` branch.

**Status: the code is complete for the base model, distillation from the teacher, feedback
crops and the update step, and has been tested on fake data only (the real teacher
soft-label files were checked against the real datasets). The base model has not been
trained on the real datasets yet, so there are no results here.**

## Scope

The student does one thing: it takes the picture of **one** item and says wet or dry.

- Input: one cropped item (BGR numpy image). The detector does the cropping.
- Output: a label and a calibrated confidence.
- Not in this folder: cropping, item counting, the "mixed load" logic, drop counting (the
  detector); asking a human and saving feedback (the bin side); the teacher; the update
  script that decides when to retrain.

## Interface

```python
from classifier import WasteClassifier

classifier = WasteClassifier(model_path="student/runs/gray/weights.pt")
result = classifier.classify_image(crop_bgr)    # numpy BGR image of ONE item
# {"label": "dry" | "wet", "confidence": 0.0-1.0,
#  "dry_prob": ..., "wet_prob": ..., "inference_ms": ...}
classifier.version                              # version name of the weights
```

`confidence` is the probability of the predicted label after temperature scaling, so
"0.8" should mean about 80% likely to be right. The human-in-the-loop threshold depends
on this. The temperature, the version name and the input size are stored inside the
weights file, so one file is enough. `classifier.version` is what the bin logs as
`student_version`.

## Where the student fits

```
camera -> detector -> student -> decision -> servo
                         |
              unsure, or every Xth item
                         v
        human labels it -> feedback folder -> teacher refits (Vibha)
                                                    v
                                 soft labels -> student retrains (update_student.py)
                                                    v
           new weights.pt  <-  calibrate -> promote.py gate
```

What the student receives and hands on:

| Hand-off | Format |
|---|---|
| Detector → student | one crop per item: uint8 BGR, the item's box + 10% padding, gray 128 outside the item |
| Student → detector | the dict above |
| Bin photos → student | `dataset_bin/{train,val,test}/{dry waste,wet waste}/<crop_id>.jpg` and `manifest.csv` (from the capture tool, `bin_vision/split_bin_dataset.py`; split by item) |
| Feedback → student | the `hitl.py` folder: `feedback.csv` and `crops/<crop_id>.jpg` (the crop the student saw, plus `human_label`) |
| Teacher → student | **soft labels, not weights**: a CSV with `split, class_folder, filename, teacher_wet_prob, teacher_logit, teacher_version`, out-of-fold for train |
| Student → bin | `weights.pt` with its version name and temperature inside |
| Student → update script | `update.json` (what went in, val numbers, temperature), for `models.csv` |

The student never asks the teacher anything while classifying, and nothing calls it except
the detector (and the update script when it retrains).

The teacher (`teacher/` on its branch, Vibha's part) is DINOv2 ViT-B/14 features plus a
logistic regression that is refit off the bin. Its notebook exports soft labels for
datasets C v2 and D v2 balanced. As of that branch these are still **in-sample** (the
teacher was fitted on the train images, so its train soft labels are almost the hard
labels) and were computed on **whole images, not crops**; both are on the teacher side's
list to change.

## Model

MobileNetV2, starting from ImageNet weights, with a 2-class head (`dry`, `wet`; the
dataset folders are `dry waste` and `wet waste`). Input 224 px by default (`--img_size 160`
trains a faster model). All layers are fine-tuned (AdamW, lr 1e-4, cosine schedule, 15
epochs, class-weighted loss, seed 42). The epoch with the best validation accuracy is kept.

## Gray and plate versions

The crop the detector hands over has either a gray or a real-plate background. The team's
hand-off table says "Proposed: gray (dataset D style); plate (dataset C) is the alternative",
and the student must be trained the way it will be used, so both are prepared. They use
the same images, seed and settings; only the background differs.

| Variant | Dataset | Background around the item | Matches a detector that sends |
|---|---|---|---|
| `gray` | `dataset_D_v2_balanced` | background removed, flat gray **128** | gray-128 crops, **the only crop `item_crop()` produces today** |
| `plate` | `dataset_C_v2` | the real plate | crops that keep the real plate pixels |

The detector's `item_crop()` sets everything outside the item's mask to 128, the same value
`remove_background.py` used, so the bin never sends a real-plate crop. The `plate` model
only matches the bin if the detector is changed to keep the plate pixels; no such option
exists in `feature/multi-item-detection` (checked: no flag or second function). Until then
`gray` is the variant that matches and `plate` is kept for the comparison. If a plate
detector appears, point `--model_path` at the matching weights; the interface is the same.
Running a model on the other variant's test set (`evaluate.py --data_dir`) measures what a
mismatch costs.

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

On the bin the student never sees a whole photo: the detector sends the item's bounding box
+ 10% padding. The public dataset images are whole photos where the item often fills only a
quarter of the frame. So every public image is first cropped to its item, using the masks in
`dataset_D_v2_balanced/masks/` (they also fit `dataset_C_v2`: same file names):

| Split | Padding around the item's box |
|---|---|
| train | random 5-30% of the box's longer side |
| val, test | exactly 10%, as the detector |

The crop is clamped to the image, like the detector's. Masks that cover under 1% of the
image are failed cut-outs; those images are used whole. `train.py` stores the mask folder
in the weights file, so `calibrate.py` and `evaluate.py` crop the same way. `--no_crop`
trains on whole images, for comparison. Bin photos are already the detector's crops, so they
are never cropped again.

## Files

| File | What it does |
|---|---|
| `common.py` | Constants (gray 128, 224 px, ImageNet normalisation), model builder, dataset loader, soft-label and feedback loaders |
| `train.py` | Fine-tunes the model (public data, bin photos, distillation, feedback crops). Uses `train/` to fit and `val/` to pick the epoch. Never reads `test/`. |
| `calibrate.py` | Temperature scaling on `val/` (a bounded search, T between 0.05 and 20). Writes T into the weights file and `temperature.json`. Prints NLL and ECE before and after. |
| `classifier.py` | `WasteClassifier` (the interface above) |
| `evaluate.py` | Test results: accuracy overall and per source (and per lighting on bin photos), confusion matrix, ECE. Can be pointed at another dataset's test set as a cross-check. `metrics()` is reused by the scripts below. |
| `update_student.py` | **One command for an update**: current weights + soft labels + feedback -> a calibrated new student |
| `promote.py` | Release gate: ship new weights only if calibrated and at least as accurate as the current ones on the bin test set |
| `experiments.py` | public-only vs bin-only vs public-then-bin, several seeds each, scored on the bin |
| `export_onnx.py` | ONNX export (calibrated probabilities as output) for TensorRT on the Nano |

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

1. **Base model.** ImageNet weights, fine-tuned on the public datasets above
   (`--variant gray`, and `plate` for the comparison).
2. **Bin photos (once `dataset_bin` exists).** Photos from the bin's own camera adapt the
   base model to the real camera, lighting and plate. Continue from the base weights:

```bash
# dataset_bin/{train,val,test}/{dry waste,wet waste}/*.jpg  (+ manifest.csv, optional)
python student/train.py     --variant bin --init_weights student/runs/gray/weights.pt --lr 3e-5 --out student/runs/bin
python student/calibrate.py --weights student/runs/bin/weights.pt     # re-fit T on bin photos
python student/evaluate.py  --weights student/runs/bin/weights.pt
```

`--init_weights` replaces the ImageNet start and keeps the starting model's input size;
the temperature is reset and must be re-fitted with `calibrate.py` (it uses the bin
photos' `val/`, so "0.8" means 80% on the bin, not on the datasets). `--lr 3e-5` is a
starting guess for fine-tuning, not tuned.
3. **Updates from feedback and the teacher** (below).

## Distillation from the teacher

The teacher does not send weights. It exports **soft labels**, its calibrated P(wet) per
image, and the student trains on them:

```bash
python student/train.py --variant gray --data_root . --soft_labels teacher/soft_labels_dataset_D_v2_balanced.csv
```

For an image with a teacher probability the loss is
`distill_alpha * CE(true label) + (1 - distill_alpha) * CE(teacher's [P(dry), P(wet)])`
(`--distill_alpha`, default 0.5; 1 = no distillation). Images without one use the true
label only. Class weights are unchanged, and choosing the best epoch always uses the true
labels, so a distilled run and a plain run are compared like for like.

- **Matching:** on (split, class folder, file name without extension), so two images with
  the same file name in different splits or classes do not collide. Both the agreed layout
  (`split, class_folder, filename, teacher_wet_prob, teacher_version`) and the layout the
  teacher notebook writes today (`path, split, label, teacher_wet_prob`) are read. On the
  real files every train, val and test image matches (gray 1902/403/411, plate 1940/416/416).
- **The current teacher CSVs are in-sample** (see above), so distillation adds little until
  the teacher exports out-of-fold train soft labels. The loader cannot check this.
- `teacher_version` is stored in the weights.

## Human feedback crops

`--feedback_dir <FeedbackStore folder>` (the folder `hitl.py` writes: `feedback.csv` and
`crops/`) adds the human-labelled crops to **train/ only**. Skipped, with the counts printed:
items the human skipped, **every item of a load whose human labels mix dry and wet** (those
loads are for the paper, not labels), missing files, and crops that are byte-for-byte copies
of a val or test image. A feedback crop takes a teacher probability if the soft-label file
has its `crop_id` (any split); otherwise it uses the human label only.

Caveat: feedback crops may show a physical item that is in the bin **test** set (another
photo of it is not caught by the duplicate check). Pin the test items
(`split_bin_dataset.py --test_items`) and keep them out of the sorting loop, or the test
accuracy will be too good.

## One update of the student

```bash
python student/update_student.py --current student/releases/student_v1/weights.pt \
    --version student_v2 --data_root . \
    --feedback_dir bin_feedback --soft_labels teacher/soft_labels_v2.csv

python student/promote.py --candidate student/runs/student_v2/weights.pt \
    --current student/releases/student_v1/weights.pt --data_dir dataset_bin     # the gate
```

`update_student.py` runs `train.py --variant bin` from the current weights (on
`dataset_bin/train` plus the feedback crops, distilling from the soft labels), then
`calibrate.py` on `dataset_bin/val`. It writes `<out>/weights.pt` (version name and
temperature inside) and `<out>/update.json` (version, date, what it replaced, images and
feedback crops used, how many had a teacher probability, teacher version, val accuracy /
NLL / ECE, temperature) for the update script's `models.csv`. Exit code 0 = a calibrated
new student exists, non-zero = failed (it refuses a version name equal to the current one,
an existing output folder without `--force`, and missing inputs). It ships nothing: the
decision is `promote.py`, on the bin test set.

Defaults: 10 epochs, lr 3e-5, `distill_alpha` 0.5, seed 42 (starting guesses, not tuned).
Without `--soft_labels` it is a plain fine-tune on the bin photos and feedback.

Rules, and how they are kept:
- **The bin test set never goes into training.** The update never reads `dataset_bin/test`
  (checked by hiding that folder: it still runs), and feedback crops that are byte-for-byte
  copies of a val or test image are dropped.
- **Every new student is calibrated before it is reported as done**, and `promote.py`
  refuses uncalibrated weights.
- **Per-version results:** `WasteClassifier.version` is what the bin logs as `student_version`.

## Versions and the release gate

Every training run has a version name (`--version`, default `<variant>-s<seed>`) stored in
the weights.

Accepted by `promote.py` only if the candidate was **calibrated**, has a **new version
name**, and its **bin test accuracy is >= the current weights'**. It then copies the
weights to `student/releases/<version>/` with an `info.json`. It also prints NLL, ECE and
the paired counts (images only the candidate gets right, and only the current gets right).
Exit code 0 accepted, 1 rejected; `--dry_run` only prints. "At least as good" means
accuracy, as agreed; with a small test set a one-image gap is noise, so read the paired
counts.

## Public vs bin data

```bash
python student/experiments.py --data_root . --public_variant gray --seeds 1 2 3
```

Trains `public_only`, `bin_only` and `public_to_bin` (`--init_weights`, lr 3e-5) for each
seed, calibrates each on `dataset_bin/val`, scores each on `dataset_bin/test`, and writes
`student/runs/experiments/summary.md` (mean and std over seeds). It needs `dataset_bin`, so
it has only been run on fake data. It is slow on a CPU: 3 trainings per seed.

## Speed on the Nano

`--img_size 160` trains at 160 px (default 224). `export_onnx.py` writes an ONNX graph whose
output is the calibrated `[dry_prob, wet_prob]`, plus a `.json` with the preprocessing to
repeat. The TensorRT FP16 build and `tegrastats` runs happen on the Nano:

```bash
trtexec --onnx=student_224.onnx --fp16 --saveEngine=student_224.engine
```

Tested (on a laptop CPU, with fake-data weights): the exported files pass the ONNX checker;
ONNX Runtime matches PyTorch to about 1e-6 at 160 and 224 px; and the preprocessing recipe
in the `.json` (RGB, bilinear resize, ImageNet mean/std) reproduces `classify_image()` (same
label on 15/15 random crops, probabilities within 1e-6). For a feel only: 160 px took 1.6 ms
and 224 px 2.9 ms per image in ONNX Runtime on this CPU. That is not a Nano number.

**Not tested:** TensorRT, FP16 and `tegrastats` (they need the Nano), and whether a model
trained at 160 px is accurate enough (it needs real training). `export_onnx.py` uses the
legacy TorchScript exporter (`dynamo=False`); PyTorch prints a deprecation warning for it.
Needs `pip install onnx` (and `onnxruntime` only if you want to repeat the checks).

## Done and not done

Done
- Training, calibration, evaluation and inference code; the interface checked against the
  detector's own `load_student()`, `item_crop()` and `make_classify()` with a dummy
  (untrained) checkpoint.
- Distillation, feedback crops, versions, `promote.py`, `experiments.py`, `--img_size`,
  `update_student.py`: all run end to end on fake data in the formats the capture-tool and
  human-in-the-loop branches write. `update_student.py` was also run with
  `dataset_bin/test` hidden and with a planted duplicate of a val image. The soft-label join
  was checked on the real teacher files and datasets (100% matched).
- A bug fixed in the temperature fit: the optimiser diverged on small or separable val sets
  (T = NaN, or 1e-5). It is now a bounded search; a warning is printed if T lands on the
  limit.
- ONNX export checked against PyTorch and ONNX Runtime.

Not done
- Real training on C v2 and D v2 balanced, and the test results. The hand-off with the
  detector needs a re-check with real weights.
- Anything on real bin photos: `dataset_bin` does not exist yet, so `experiments.py` and
  the bin fine-tune have not run on real data.
- The gray vs plate decision (and whether the detector will ever send plate crops).
- Out-of-fold, crop-based soft labels from the teacher (teacher side).
- TensorRT, FP16 and power measurements on the Nano.
- The update script that decides when to retrain (every K labels), keeps `models.csv`, and
  copies weights to the bin; it will call `update_student.py` and `promote.py`.
- Handing over the `.pt` files (`*.pt` is gitignored; they will go in a release).

## Notes

- Nothing here comes from the `ai_dustbin_2` repo.
- Data is non-commercial (RealWaste is CC BY-NC-SA 4.0); see the main README.
