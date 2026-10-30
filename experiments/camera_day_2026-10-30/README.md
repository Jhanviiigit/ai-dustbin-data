# Camera day 2026-10-30: real-camera test, one loop round, detector

The first run of the whole system on photos from the bin's own camera: background removal, scoring of the
student and the teacher, one update round (feedback → ingest → teacher → student → gate → deploy), and the
detector's SEPARATE rule. Code on `main` at d41c64c; nothing in the shared code was changed.

Photos, crops and model files are in release **`data-camera-day-2026-10-30`** (see [Files](#files)).

## Results at a glance

Real-camera test pile: **113 photos of 58 items** no model trained on. Full tables: [`results/scores_tables.md`](results/scores_tables.md).

| Model | Phone photos (test_fixed, 72) | Real camera, per photo [95% CI] | Real camera, per item (58) | AUROC | ECE |
|---|---|---|---|---|---|
| Student `organik_gray_v2` (baseline) | 90.3% | **81.4%** [73.3, 87.5] | 84.5% | 0.921 | 0.100 |
| Teacher `teacher_v1_gray` | 95.8% (reported earlier) | **93.8%** [87.8, 97.0] | 96.6% | 0.991 | 0.050 |
| Student `student_u1` (after one loop round) | 95.8% | **91.2%** [84.5, 95.1] | 93.1% | 0.971 | 0.071 |
| Teacher `teacher_v2` (after the loop) | 95.8% | **93.8%** [87.8, 97.0] | 96.6% | 0.994 | 0.044 |
| Variant B: student, bin labels only, no teacher | not gated | **92.0%** [85.6, 95.8] | 94.8% | 0.966 | 0.061 |

- **Domain gap.** Student 90.3% → 81.4% from phone photos to the bin camera; teacher 95.8% → 93.8%. The student's
  errors are mostly wet called dry (13 of 21; wet recall 0.76).
- **Distillation gap on the camera.** Teacher 93.8% vs student 81.4%: student alone wrong on 19 photos, teacher
  alone on 5, exact McNemar p 0.007.
- **Does the loop help.** Student 81.4% → 91.2% after one round. Per photo: 13 fixed, 2 broken, p 0.007.
  Per item (the safer unit: two photos of one item are not independent): 5 fixed, 0 broken, **p 0.06**.
  After the round the student is level with the teacher (p 0.63).
- **Where the gain comes from.** Variant B (the same update with the bin labels only, no teacher soft labels)
  reaches 92.0%, the same as the full loop (p 1.0; per item 6 fixed, 0 broken, p 0.03). On accuracy the gain is
  from the bin-camera labels. The teacher's effect is on confidence: with it the student sends **9%** of loads to a
  person, without it **19%**.
- **Teacher update.** 21 bin crops against ~1,900 public images do not move the teacher (v1 vs v2: one photo
  each way, p 1.0).

Human effort at the 0.8 threshold (per photo):

| Model | Sent to a person | Errors caught | Wrong auto-tilts left | Accuracy of what the bin does alone |
|---|---|---|---|---|
| Baseline student | 16% (18/113) | 11 of 21 | 10 | 89.5% |
| student_u1 | 9% (10/113) | 5 of 10 | 5 | 95.1% |
| Variant B | 19% (22/113) | 6 of 9 | 3 | 96.7% |

Detector, on 234 single-item photos: the SEPARATE rule at **0.09 (now) wrongly stops 26.9%** of single items;
0.25 (PR #18) 9.8%; **0.40: 3.8%** (under the 5% target).

## What was done, step by step

### 1. Photos

Taken on the bin plate with the bin camera (640×480, `nvcamtest_*` files): 126 in `dry/`, 115 in `wet/`, plus 8
empty-plate photos in `background/`. Labels are the folder the photo was put in. **Agreed rule: dry items with
food on them stay dry** (e.g. the plastic container, black bowl, wooden spoon).

### 2. Items and the two piles (`scripts/make_piles.py`)

The camera files are long numbered bursts (one session has 118 photos), so the filename does not say where one
item ends. Every photo was looked at on contact sheets and consecutive shots of the same item grouped (usually 2:
the item turned over). Where unsure, photos were put in one item (merging is safe; splitting could leak one item
into both piles). Items cut from one thing (omelette pieces, a cabbage and its leaves, chocolate pieces, orange
peel and segments) share a `source` and stay in one pile.

Split by item, seed 42: items with 1 photo → test (test wants 1-2), items with 3+ photos or a shared source →
loop, the rest at random to 30 wet + 30 dry in test.

| Pile | Items (wet / dry) | Photos | Used for |
|---|---|---|---|
| Test | 60 (30 / 30) | 117 | scoring only |
| Loop | 57 (26 / 31) | 124 | replayed through the bin as feedback |

No item in both piles (checked). Item list: `results/items.csv`; test items: `results/test_items.txt`.

### 3. Background removal (`scripts/seg.py`, `scripts/crop2.py`)

The bin's own detector: `bin_vision/multi_item.py` `ItemDetector` (empty-plate subtraction in Lab, shadows
down-weighted) and `item_crop` (gray 128 outside the item's mask, as the student was trained). Three settings
were changed for these photos, all in `seg.py`, not in the bin's code:

| Setting | Bin default | Used | Why |
|---|---|---|---|
| `diff_threshold` | 28 | **16** | at 28, 29 of 241 photos had no item found: low-contrast items on the wooden plate |
| Empty-plate reference | one | the best of the 8 plate photos, matched on the plate border | the plate photos were taken at different times of the day |
| Thin blobs touching the left/right frame edge (< 12% of the width) | kept | dropped | the plate edge shifts slightly between shots and shows up as a strip |
| Item | each blob separate | largest blob + fragments within 8% of the frame width | one item per photo; pale food breaks into fragments |

How 16 was chosen (`scripts/sweep2.py`): each empty-plate photo was run against the other 7 as if it were a
load; at every threshold from 14 to 28 no part of an empty plate was flagged once edge strips are dropped, and
the number of photos with no item fell from 29 (28) to 1 (16). 14 was not lower in misses. Every crop was then
checked by eye on overlay sheets. Transparent items are only partly masked (the clear container keeps mostly its
food; the chapati is patchy), as on the live bin.

Not cropped (no item found), 7 photos: test: `biskut.jpg` (taken in a different setup, no matching empty plate),
the cinnamon stick (both photos), one cashew photo; loop: a tiny purple item (both photos, below the detector's
0.4% minimum area), one chocolate photo. Scored test set: 113 photos of 58 items.

### 4. Models scored

- **Student `organik_gray_v2`**: release `student-organik_gray_v2`, the deployed baseline (T 1.2247).
- **Teacher `teacher_v1_gray`**: refit here with the repo's code and data, because the earlier joblib was not on
  this machine: `python teacher/teacher.py --version teacher_v1_gray --gray dataset_D_v2_balanced` (release
  `data-v2`). DINOv2 ViT-B/14 + logistic regression, C 0.3, T 0.964; 99.3% on the D v2 test split (411).
- `student_u1`, `teacher_v2`, variant B: from the loop (step 6).

Each model classifies each crop through its `classify_image` (`scripts/predict.py`); one CSV per model in
`results/predictions/` with the wet probability of every photo.

### 5. Metrics (`scripts/report.py`, `update/metrics.py`)

Per photo and per item (an item's wet probability = mean over its photos): accuracy with a 95% Wilson
interval, balanced accuracy, per-class precision / recall / F1, confusion matrix, AUROC, ECE (10 bins). Review
policy at 0.8: share sent to a person, errors caught, accuracy of what the bin does alone. Any two models
compared on the same photos with the exact McNemar test.

### 6. One loop round (`scripts/replay.py`, then the update pipeline)

On copies of the seeded pool (`pool-v1`), `releases/` and `deploy/` (baseline deployed), so the team's real pool
was not changed.

1. **Bin replay.** Each loop photo, in shooting order, is one load through the bin's code: detector (step 3) →
   `assess` (student + SEPARATE rule, max_area_frac 0.40, see step 7) → `ReviewPolicy` (below 0.8 → person;
   every 10th auto-tilt → audit) → `FeedbackStore`. The person's answer is the folder label.
   Result on 124 photos: 54 tilt dry, 39 tilt wet, 25 ask a person, 3 SEPARATE, 3 no item. **34 loads labelled**
   (25 low confidence, 9 audits); 14 wrong auto-tilts no person saw. Files: `results/loop/feedback.csv`, `loads.csv`.
2. **Ingest** (`update/ingest.py`, then `update/run_update.py --pool pool --feedback feedback`): 34 rows → train 17,
   val 3, test_audit 9, test_hard 5; no audit leaks, no near-copies. **Trigger: go, without `--force`** (17 new
   train groups, need 15; dry 4, wet 13). 171 augmented dry copies to balance train. Then exit 3: waiting for the
   teacher. (`results/loop/update_1.json`)
3. **Teacher** (`teacher/pool_soft_labels.py --pool pool --version teacher_v2 --gray dataset_D_v2_balanced`):
   fitted on 1,902 public + 21 feedback train crops, **never on Organik**; leak guard 0 of 1,923; C 0.1, T 0.868;
   val 99.0%, test_fixed 95.8%; 690 soft labels, `teacher_wet_prob = sigmoid(logit / T)` (PR #21).
4. **Student** (`run_update.py` again → `student/update_student.py`): starts from the baseline, 10 epochs, lr 3e-5,
   distill_alpha 0.5, on 690 train images (502 Organik, 17 feedback, 171 augmented), every one with a teacher
   probability. Best val 98.7% (75 images). Calibrated on val: T 0.363. Threshold: none from val (too small for
   the rule), so the 0.8 fallback stays.
5. **Gate** (`student/promote.py`): **ACCEPTED.** test_fixed 3 errors vs 7 (only candidate wrong 1, only
   baseline wrong 5, McNemar p 0.22); test_audit 9/9 and test_hard 5/5 reported (test_audit gated from 30).
   (`results/loop/gate.json`)
6. **Deploy** `deploy/current` = student_u1, threshold 0.8; `models.csv` row `update_1`
   (`results/loop/models.csv`).
7. **Variant B** (for the paper's comparison): the same training run with `--distill_alpha 0` (no soft labels),
   from the same baseline and pool, calibrated the same way (T 1.467). Not gated or deployed; scored only.

Caveat: feedback is grouped by load, and here each photo is its own load, so two photos of one item can sit in
different pool splits (e.g. train and test_audit). That can inflate test_audit and val (and so the epoch choice
and T), not the test pile or test_fixed. On a live bin an item is not normally dropped twice.

### 7. Detector: SEPARATE rule (`scripts/detector.py`)

`blob_features` + `suspect_multiple` from `bin_vision/multi_item.py` on the 234 single-item photos (masks from
step 3). Area fraction: median 0.041, 95th percentile 0.31, max 1.0 (the pink lid fills the frame).

| max_area_frac | Single items wrongly stopped | by area | by notch rule |
|---|---|---|---|
| 0.09 (now) | 63 / 234 = 26.9% | 59 | 4 |
| 0.15 | 40 / 234 = 17.1% | 35 | 5 |
| 0.25 (PR #18) | 23 / 234 = 9.8% | 18 | 5 |
| 0.30 | 18 / 234 = 7.7% | 13 | 5 |
| **0.40** | **9 / 234 = 3.8%** | 4 | 5 |
| 0.50 | 7 / 234 = 3.0% | 2 | 5 |

`DEFAULT_RULE` in `multi_item.py` was **not** changed here. No touching-items loads were photographed, so how
many merged loads each value still catches cannot be measured from this set.

## Limits

- 58 test items from one session, one plate, one camera position: an interval of about ±6-10 points; small
  differences between models are noise. Report per-item p-values alongside per-photo ones.
- Labels are one person's folder labels; ambiguous items follow the agreed rule (food-soiled dry stays dry).
- The camera height was not recorded with the photos; note it with any max_area_frac value.
- The loop pile and the test pile come from the same session, plate and lighting, so the loop's gain is
  adaptation to this camera setup; a second session would show how far it carries.
- teacher_v1_gray was refit here; its 95.8% phone-photo figure is the one reported earlier, not re-measured.
- One loop round only.

## Reproduce

```bash
# photos: unzip camera_day_photos.zip from release data-camera-day-2026-10-30 -> bin_piles_2026-10-30/, background/
S=experiments/camera_day_2026-10-30/scripts
python -P $S/crop2.py . bin_piles_2026-10-30 background 16 crops            # step 3 (needs opencv)
python -P $S/sweep2.py . bin_piles_2026-10-30 background                    # threshold sweep
python -P $S/predict.py . crops test <weights.pt | teacher.joblib> <name> preds/test_<name>.csv   # step 4
python -P $S/report.py . preds scores_tables.md                             # step 5
python -P $S/replay.py . bin_piles_2026-10-30 background crops <deploy> <feedback> 0.40          # step 6.1
python update/run_update.py --pool <pool> --feedback <feedback> --releases <releases> --deploy <deploy> --models_csv <models.csv> --runs <runs>
python teacher/pool_soft_labels.py --pool <pool> --version teacher_v2 --gray dataset_D_v2_balanced --out <runs>/teacher_v2/soft_labels.csv
python update/run_update.py --pool <pool> --releases <releases> --deploy <deploy> --models_csv <models.csv> --runs <runs>
python -P $S/detector.py . bin_piles_2026-10-30 background crops            # step 7
```

`make_piles.py` records how the piles were made: it reads the original `dry/dry/`, `wet/wet/` folders and its
item groups index the sorted file list of each folder. The release holds the piles it produced; each file keeps
its original name after the item id (`<item_id>__<original name>.jpg`), and `items.csv` maps both ways.

## Files

In this folder:

| Path | What |
|---|---|
| `results/scores_tables.md` | every metric, per photo and per item, review policy, paired tests |
| `results/predictions/test_*.csv` | each model's wet probability for every test photo |
| `results/items.csv`, `test_items.txt` | photo → item, label, pile, source |
| `results/crops.csv` | photo → crop, empty-plate reference used, area fraction |
| `results/loop/` | feedback.csv, loads.csv, update_1.json, teacher_v2 soft labels, gate.json, update.json, models.csv, logs |
| `scripts/` | piles, segmentation, threshold sweep, prediction, report, replay, detector |

In release `data-camera-day-2026-10-30`:

| Asset | What |
|---|---|
| `camera_day_photos.zip` | the original photos as two piles (`test/`, `loop/`, by class) + `background/` + item lists |
| `camera_day_crops.zip` | every crop (gray outside the mask) + crops.csv |
| `camera_day_feedback.zip` | the replayed bin feedback folder (crops, frames, feedback.csv, loads.csv) |
| `student_u1.pt` | the accepted student after the loop (T 0.363) |
| `student_u1_B.pt` | variant B (no teacher) |
| `teacher_v1_gray.joblib`, `teacher_v2.joblib` | the two teachers |
