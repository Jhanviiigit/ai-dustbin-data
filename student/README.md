# Student model

The student is the small, fast wet/dry classifier that runs on the bin. This folder holds
its training, calibration, evaluation, inference and update code. It is built on the
`student` branch.

**Status: the code is complete for the base model, distillation from the teacher, feedback
crops and the update step. The base model is trained on the Organik gray crops only (see
"Current training" below), with cross-validated results. Distillation, feedback crops and the
update step have been tested on fake data only (the real teacher soft-label files were
checked against the real datasets). Nothing has run on real bin photos yet.**

## Which data to use for what

| Purpose | Use | Where |
|---|---|---|
| **Train the student** | `dataset_organik_v2/train` (502 images) | release `data-organik-v2` |
| Pick the best epoch | `dataset_organik_v2/val` (72) | same |
| **Check the pipeline** | **`dataset_organik_v2/test`** (72 images, 36 dry and 36 wet; the deployed model never trained on it). Not the headline number: 72 images is too few (5 of the 7 errors are one straw). The headline number is the 5-fold cross-validation result below | same |
| The trained student | `weights.pt`, version `organik_gray_v2` | release `student-organik_gray_v2` |
| The teacher's training data | Roboflow and RealWaste (`dataset_C_v2`, `dataset_D_v2_balanced`) | release `data-v2`; **not for the student** |

Do not use: `data-organik-v1` (uncurated and not split), `data-v1` (includes `dataset2`, whose
origin is unknown), or `dataset_C_v2` / `dataset_D_v2_balanced` to train or test the student
(the sections below that mention them describe the earlier public-data runs). Details,
download commands and results: "Current training" below.

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

classifier = WasteClassifier(model_path="student/runs/organik/weights.pt")
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
| Feedback → student | the `hitl.py` folder: `feedback.csv` and `crops/<crop_id>.jpg` (the crop the student classified and the human was shown, `human_label`, and `from_drop`: 1 = cut when the item landed, 0 = cut from the final frame) |
| Teacher → student | **soft labels, not weights**: a CSV with `split, class_folder, filename, teacher_wet_prob, teacher_logit, teacher_version`, out-of-fold for train |
| Student → bin | `weights.pt` with its version name and temperature inside |
| Student → update script | `update.json` (what went in, val numbers, temperature), for `models.csv` |

The student never asks the teacher anything while classifying, and nothing calls it except
the detector (and the update script when it retrains).

The teacher (`teacher/` on its branch, Vibha's part) is DINOv2 ViT-B/14 features plus a
logistic regression that is refit off the bin. As of its latest commit it writes the agreed
soft-label format, with **out-of-fold** train labels (checked: it is now wrong on 32 of 1,902
gray and 12 of 1,940 plate train images, where its earlier in-sample files were wrong on
none), and per its README it frames crops with this folder's `common.py` (`crop_to_item`,
the paddings, the preprocessing), so teacher and student see identical crops. That makes
`student/common.py` a shared file: the teacher imports names from it, so renaming or
removing them breaks the teacher.

## Model

MobileNetV2, starting from ImageNet weights, with a 2-class head (`dry`, `wet`; the
dataset folders are `dry waste` and `wet waste`). Input 224 px by default (`--img_size 160`
trains a faster model). All layers are fine-tuned (AdamW, lr 1e-4, cosine schedule, 15
epochs, class-weighted loss, seed 42). The epoch with the best validation accuracy is kept.

## Gray and plate versions

*Earlier public-data comparison (Roboflow + RealWaste). The student is now trained on the
Organik gray crops only (see "Current training"); the plate version is not planned.*

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
python bin_vision/multi_item.py --webcam --student_dir student --model_path student/runs/organik/weights.pt
```

## Item crops (same framing as the bin)

*This applies to the public datasets. The Organik crops are already cut to the item with the
detector's framing, so they train with `--variant bin` and no masks.*

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

**For the Organik student, use the commands under "Current training" below.** The block here
is the earlier public-data run (Roboflow + RealWaste), kept for the comparison.

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

## Current training: Organik gray crops only

Decision (2026-10-28): the student is trained **only** on the Organik / Anorganik photos
(release `data-organik-v2`: background removed, gray 128, cut to the item), starting from
ImageNet weights. The public datasets (Roboflow, RealWaste) are for the teacher alone. Gray
only for now; a raw-background version is decided later by accuracy. Organic is `wet waste`,
Inorganic is `dry waste`. The crops already have the detector's framing, so they train with
`--variant bin` (no masks, no cropping).

**Get the data and the model** (private repo: you need access; `gh auth login` first):

```bash
# the split the student is trained on (646 images; train 502, val 72, test 72)
gh release download data-organik-v2 --repo Jhanviiigit/ai-dustbin-data --pattern "dataset_organik_v2.zip"
unzip dataset_organik_v2.zip
# the trained student (organik_gray_v2, temperature 1.2247 inside the file)
gh release download student-organik_gray_v2 --repo Jhanviiigit/ai-dustbin-data --pattern "weights.pt" --dir student/runs/organik
```

The same data release also holds `organik_gray_curated.zip`, the 672 source crops, if you want
to rebuild the split. This is Sian's hand-curated copy of `data-organik-v1`: pixels are
identical, but 9 images were moved from Organic to Inorganic
(`20250922_143702`, `_143714`, `_143724`, `_143747`, `_143757`, `_143813`,
`IMG_20250922_143613`, `_143616`, `_143619`) and 5 Organic images were deleted
(`1761117174238`, `…261`, `…282`, `…303`, `20250922_143647`).

Source of the photos: Mendeley Data, "Dataset waste image classification Organik and
Inorganic", contributor kevin bagus, Version 1, 5 November 2025, DOI
[10.17632/t2sy8k4jpj.1](https://data.mendeley.com/datasets/t2sy8k4jpj/1). Smartphone photos
taken top-down on the Primakara University campus, Denpasar, Bali. Licence **CC BY 4.0**:
credit the author and indicate changes (here: background removed, cropped, re-labelled and
curated). Cite it in the paper.

**Rebuild the dataset** (for reference only). **The released `dataset_organik_v2.zip` is the
official split: use it, do not rebuild.** The rebuild is not bit-reproducible across
machines: on another machine `regroup_organik.py` gave 80 groups instead of 82 and split 127
images differently, because 166 image pairs sit right at the 0.18 cut and tiny numerical
differences tip them. Two steps; use the second one's output:

```bash
python data_prep/split_organik.py   --src organik            --out dataset_organik      # hash + file-order groups; drops an item filed under both classes
python data_prep/regroup_organik.py --src dataset_organik    --out dataset_organik_v2   # groups by image similarity and re-splits
```

The first split leaked: the same item photographed in another pose got a different group and
landed on both sides, which inflated the scores (5-fold 97.7%, and 8 of 78 test images had a
near-copy in train). `regroup_organik.py` clusters images by ImageNet-feature similarity
(within each class) and splits whole clusters, so no val or test image has a train image at
similarity >= 0.90. That is a safeguard, not a proof: the data has no item IDs. v2 split:
train 502 (170 dry, 332 wet), val 72 and test 72 (36 + 36 each). The loss is class-weighted,
so train keeps all of the larger class.

**Train** (uses the GPU when CUDA is available; a few minutes on a Colab T4):

```bash
python student/train.py --variant bin --data_dir dataset_organik_v2 --version organik_gray_v2 --out student/runs/organik
```

Do **not** run `calibrate.py` on this val set. Every val image is classified correctly, so the
fit runs to the lower limit (T = 0.05) and makes every prediction look certain (on the first,
leaky split this raised the test NLL from 0.08 to 0.69). Use the temperature from
cross-validation instead:
**T = 1.2247**, written into the weights file (`ckpt["temperature"]`, `ckpt["calibrated"] = True`).

**Results** (5-fold cross-validation on all 646 images, folds split by cluster, run in
Colab; the notebook is not in this repo). Each fold trains from ImageNet on the other four,
a slice of them picks the epoch, and the held-out fold is scored:

| | Result |
|---|---|
| Accuracy over the 5 folds | 96.75%, std 1.4 (folds 94.6% to 98.4%), 21 errors of 646 |
| dry | precision 0.951, recall 0.963, F1 0.957 |
| wet | precision 0.978, recall 0.970, F1 0.974 |
| AUC-ROC | 0.994 |
| Temperature fitted on the pooled out-of-fold predictions | 1.22; ECE 0.017 -> 0.008 (cross-fitted 0.008), NLL unchanged |

Freezing the first N modules of `model.features` (same folds): N = 0 96.75% (21 errors),
N = 10 96.44% (23), N = 15 97.22% (18), N = 19 (head only) 93.35% (43). 0, 10 and 15 are
within noise, so everything is fine-tuned.

The single deployed model (`organik_gray_v2`, epoch 6) scored 90.3% on its 72 test images
(1 wet and 6 dry errors; 5 of the 7 are one green plastic straw read as a leaf, so the colour
leans towards wet). Its wrong answers had confidence 0.53 to 0.74 and the correct ones a
median of 0.99, so a human-review threshold near 0.8 would have caught all 7 (and sent 15% of
items to a person). That is read off 72 images; do not tune on it. 72 images is too few for
a headline number: quote the cross-validation.

## Training stages

1. **Base model.** ImageNet weights, fine-tuned on the Organik crops (above). The
   `gray` / `plate` runs on the public datasets below are the teacher's data and are only
   kept for comparison.
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
image, and the student trains on them. The existing soft-label files cover the public
datasets only; the Organik images need a new file from the teacher before the student can
distil on them (the deployed `organik_gray_v2` was trained on the true labels only):

```bash
python student/train.py --variant gray --data_root .     --soft_labels teacher/soft_labels_dataset_D_v2_balanced_teacher_v1_gray.csv
```

For an image with a teacher probability the loss is
`distill_alpha * CE(true label) + (1 - distill_alpha) * CE(teacher's [P(dry), P(wet)])`
(`--distill_alpha`, default 0.5; 1 = no distillation). Images without one use the true
label only. Class weights are unchanged, and choosing the best epoch always uses the true
labels, so a distilled run and a plain run are compared like for like.

- **Matching:** on (split, class folder, file name without extension), so two images with
  the same file name in different splits or classes do not collide. Both the agreed layout
  (`split, class_folder, filename, teacher_wet_prob, teacher_version`) and the older layout
  of the teacher notebook (`path, split, label, teacher_wet_prob`) are read. On the latest
  real files every train, val and test image matches (gray 1902/403/411, plate 1940/416/416).
- **Out-of-fold matters:** soft labels from a teacher that saw the image add nothing over the
  true label. The latest teacher files are out-of-fold, but most are still very confident
  (91% of gray and 96% of plate train labels are above 0.99), so distillation mostly agrees
  with the true labels and differs on the few the teacher gets wrong or is unsure about. The
  loader cannot check out-of-fold-ness itself.
- `teacher_version` is stored in the weights.

## Human feedback crops

`--feedback_dir <FeedbackStore folder>` (the folder `hitl.py` writes: `feedback.csv` and
`crops/`) adds the human-labelled crops to **train/ only**. Every item the human labelled is
used, **including the items of loads they called dry+wet**: each crop was labelled on its
own, and those are the loads the student was unsure about, so they are among the most
useful examples. Skipped, with the counts printed:
- items the human skipped (empty `human_label`), and missing files;
- a crop whose `box` **overlaps another item's box in the same load** (touching items: the
  crop may contain part of its neighbour, so its label is not reliable). Boxes that only
  touch along an edge do not count, and the other item counts even if the human skipped it.
  A box that cannot be read is kept and counted. **Exception: `from_drop = 1`.** Those crops
  are the ones saved at the moment the item landed (the crop the student classified and the
  human was shown), while it was still alone, so a later item landing on top cannot have
  contaminated them: they are kept even when their boxes overlap, and the printout counts
  them ("overlapping but kept"). A `feedback.csv` without a `from_drop` column treats every
  crop as `from_drop = 0`;
- a crop that is a byte-for-byte copy of a val or test image.

A feedback crop takes a teacher probability if the soft-label file has its `crop_id` (any
split); otherwise it uses the human label only.

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
- Training on C v2 and D v2 balanced (not planned: the student uses Organik only). The
  hand-off with the detector needs a re-check with the real `organik_gray_v2` weights.
- Anything on real bin photos: `dataset_bin` does not exist yet, so `experiments.py` and
  the bin fine-tune have not run on real data.
- The gray vs plate decision (and whether the detector will ever send plate crops).
- TensorRT, FP16 and power measurements on the Nano.
- The update script that decides when to retrain (every K labels), keeps `models.csv`, and
  copies weights to the bin; it will call `update_student.py` and `promote.py`.
- Handing over the `.pt` files (`*.pt` is gitignored; they will go in a release).

## Notes

- Nothing here comes from the `ai_dustbin_2` repo.
- Data is non-commercial (RealWaste is CC BY-NC-SA 4.0); see the main README.
