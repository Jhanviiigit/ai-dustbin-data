# Teacher model

Frozen **DINOv2 ViT-B/14** features + **logistic regression**, temperature-scaled. It runs off the bin, learns from
human feedback, and hands soft labels to the student (MobileNetV2) for distillation.

- `teacher.py`: the module (indexing, features, fit/refit, out-of-fold soft labels, evaluation, save/load, CLI).
- `teacher_model.ipynb`: Colab driver: per-background teachers, harder evaluation, soft labels, update cycle.

It uses `crop_to_item()`, the padding and the preprocessing from `student/common.py`, so teacher and student see
identical crops. Bin crops (`dataset_bin`, feedback) are already the detector's crops and are not cropped again.

## Formats it reads / writes
| # | | Format |
|---|---|---|
| 2 | reads | `dataset_bin/{train,val,test}/{dry waste,wet waste}/<crop_id>.jpg` + `manifest.csv` (split by `item_id`) |
| 4 | reads | hitl.py feedback folder: `feedback.csv` + `crops/<crop_id>.jpg` (via `crop_file`). Selected by `student/common.py` `load_feedback()`: every human-labelled item is kept (dry+wet loads too); only crops whose box overlaps another item in the same load, unlabelled rows, and copies of val/test images are dropped |
| 5 | writes | `soft_labels_<dataset>_<version>.csv`: `split, class_folder, filename, teacher_wet_prob, teacher_logit, teacher_version` |
| 6 | writes | `<version>.joblib`: classifier as plain arrays (scaler mean/scale, weights, intercept: loads on any scikit-learn version), C, temperature, backbone, crop settings, version, trained_on |

- Manifest rows are matched on **split + class folder + file stem**, using `new_path` with `\` turned into `/`.
- `teacher_logit` = raw log-odds of wet. `teacher_wet_prob = sigmoid(logit / T)`, calibrated on val.
  For distillation at temperature τ use `sigmoid(teacher_logit / τ)`.
- Train rows are **out-of-fold** (5 folds by group: manifest duplicate group / bin `item_id` / feedback `load_id`).
  A photo shared by C and D falls in the same group.
- `classify_image(crop_bgr)` returns the same dict as the student: `label` (`dry`/`wet`), `confidence`, `dry_prob`, `wet_prob`, `inference_ms`.

## Update cycle (after every K human labels)
```
python teacher/teacher.py --version teacher_v2 \
    --gray data/dataset_D_v2_balanced --bin data/dataset_bin --feedback data/feedback --out teacher_out
```
Refits on old data + new feedback (never new only), picks C and T on the val splits, and writes format 5 per
dataset (+ feedback), **one combined `soft_labels_<version>.csv` for `update_student.py --soft_labels`**, and format 6. Sian then distils, calibrates, and ships the student only if it is at least
as good on the bin test set.
