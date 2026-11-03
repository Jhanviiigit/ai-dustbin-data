# Session 2 (2026-11-02): a fresh real-camera test set

A second camera session, used **only as a test set**. Fixed models are scored on new items that nothing has
seen; no model is trained or tuned on these photos. Same detector settings and metrics as session 1
([`../camera_day_2026-10-30/`](../camera_day_2026-10-30/README.md)), whose scripts are reused unchanged.

**Why.** Session 1's loop pile and test pile came from one session, plate and lighting, and its test pile has
been studied a lot. Session 2 checks whether the results hold on new items, a new plate and new lighting.

| Question | Session 2 answers it with |
|---|---|
| Does the loop's gain hold on new items? | student_u1 vs the baseline student |
| Does the domain gap reappear? | the baseline here vs its 90.3% on phone photos |
| Is the teacher still the most robust model? | teacher_v1_gray / teacher_v2 vs the students |
| Are student_u1 and variant B still tied? | student_u1 vs student_u1_B |

## Status

| Part | State |
|---|---|
| Dry: 80 photos, **27 items** | grouped, cropped, every crop checked by eye |
| Wet: 69 photos, **23 items** | grouped, cropped, every crop checked by eye |
| Empty plate: 10 photos | in; the wet photos show the same plate and light |
| Camera height | 35-40 cm |
| Scored | 50 items, 148 of 149 photos (one photo with no item found) |

Dry notes: the cardboard box is shot closed, then open (one item); photos 69-70 show a glue stick lying on a
ruler (two items in one shot, both dry; kept as one item). The plate is a lighter wood than on 30 Oct; the
detector found an item in all 80 dry photos at the 30 Oct settings. As in session 1, transparent things (the
bottle, the ziplock bag, the keyboard cover) and flat cardboard are only partly masked.

Wet notes: items were grouped by eye on contact sheets, merging when unsure. Merged on that basis: wet photos
0-2 (a bell pepper, cut), 26-28 (green chilli pieces), 29-31 (cucumber slices) and 32-34 (seeded cucumber wedges).
The whole radish (55-57) and the cut radish piece (67-68) are one item. Names given in `results/items.csv`
notes: yam (41-43), grated carrot (44-46). Pale items (the yam, the potato) are only partly masked.

## Results

Fixed models scored on the 50 new items, with the session 1 detector settings (threshold 16, best-matching
plate photo, edge strips dropped) and no training or tuning on these photos. Full tables:
[`results/scores_tables.md`](results/scores_tables.md).

| Model | Per photo (148) [95% CI] | Per item (50) | Photo errors | AUROC | ECE |
|---|---|---|---|---|---|
| Student `organik_gray_v2` (baseline) | 86.5% [80.0, 91.1] | 92.0% | 20 | 0.952 | 0.085 |
| Student `student_u1` (after the 30 Oct loop round) | 91.9% [86.4, 95.3] | 94.0% | 12 | 0.972 | 0.082 |
| Variant B: bin labels only, no teacher | 91.2% [85.6, 94.8] | 92.0% | 13 | 0.969 | 0.040 |
| Teacher `teacher_v1_gray` | 91.9% [86.4, 95.3] | 94.0% | 12 | 0.956 | 0.057 |
| Teacher `teacher_v2` | 93.2% [88.0, 96.3] | 96.0% | 10 | 0.962 | 0.052 |

**Detector misses.** No item found: `nvcamtest_8159_s00_00019.jpg` (a tiny green piece, item `s2_wet_007`); it is
left out of the scores. Wrong crop: `nvcamtest_8159_s00_00022.jpg` (the pale piece, item `s2_wet_008`): the box is
on a shadow patch and the item is not masked. It stays in the scores; without it each model's per-photo accuracy
changes by less than 1.5 points (baseline 87.1%, student_u1 91.8%, variant B 91.2%, teacher_v1 92.5%,
teacher_v2 93.9%).

### The four questions (per-item p-values; per-photo in brackets)

1. **Does the loop's gain hold on new items?** The direction is the same as on 30 Oct, but it is not significant.
   `student_u1` vs the baseline: per item 94.0% vs 92.0%, 3 items only the baseline wrong, 2 only `student_u1`
   wrong, **p = 1.0** (per photo 91.9% vs 86.5%, 11 fixed, 3 broken, p = 0.057).
2. **Does the domain gap reappear?** Only partly. The baseline scores 86.5% per photo here [80.0, 91.1], against
   90.3% [81.3, 95.2] on the 72 phone photos and 81.4% in session 1. The gap is about 4 points and the intervals
   overlap.
3. **Is the teacher still the most robust model?** Not clearly. `teacher_v1_gray` ties `student_u1` (per item
   p = 1.0; per photo 7 vs 7, p = 1.0). `teacher_v2` vs the baseline: per item p = 0.625 (per photo p = 0.031).
   The errors differ in kind: the students mostly call dry items wet (`student_u1` 9 of its 12 photo errors), the
   teachers mostly call wet items dry (`teacher_v1_gray` 7 of 12).
4. **Are `student_u1` and variant B still tied?** Yes: per item 0 vs 1, p = 1.0 (per photo 2 vs 3, p = 1.0).

Review policy at 0.8, per photo:

| Model | Sent to a person | Errors caught | Wrong auto-tilts left | Accuracy of what the bin does alone |
|---|---|---|---|---|
| Baseline | 11% (16/148) | 8 of 20 | 12 | 90.9% |
| student_u1 | 7% (10/148) | 3 of 12 | 9 | 93.5% |
| Variant B | 10% (15/148) | 5 of 13 | 8 | 94.0% |
| teacher_v1_gray | 10% (15/148) | 7 of 12 | 5 | 96.2% |
| teacher_v2 | 9% (14/148) | 6 of 10 | 4 | 97.0% |

The models have different calibration temperatures, so the shares sent to a person are not like for like.

### Reading the results

- Session 2 neither confirms nor contradicts the loop's gain from 30 Oct: the direction repeats, the size is
  smaller, and no per-item test is significant.
- The baseline does better on this plate than in session 1, so there is less error left for the loop round to
  remove.
- `student_u1` and variant B are indistinguishable again, so a benefit of the teacher's soft labels on accuracy
  is still not detected.
- 50 items give 95% intervals of about +/-6 points per photo and +/-8 per item; differences of one or two items
  are noise.

## Limits

- 50 items from one session, one camera and one plate; no touching-items loads.
- Labels are one person's folder labels; the item groups were made by eye, with merges where unsure.
- The camera height is given as a range (35-40 cm).
- The photos and crops are not in the repo yet (release `data-session2-2026-11-02` is to be created).

## Steps to reproduce

Run everything from the repo root, on branch `session2` (or `main` once it is merged). Needs Python with
`opencv-python`, `torch`, `torchvision`, `scikit-learn`, `pandas`, `Pillow` (as for `student/` and `teacher/`).
CPU is enough: nothing trains; the teachers download DINOv2 (~330 MB) once.

### 1. The session folder

```
session2/
  dry/          80 photos     (from session2_dry_background.zip)
  background/   10 photos     (same zip)
  wet/          the wet photos, straight from the camera (not WhatsApp copies: compression changes the images)
```

Unzip `session2_dry_background.zip` (from Jhanvi) so it gives `session2/dry` and `session2/background`, then
put the wet `.jpg` files in `session2/wet/`. Check: open `wet/` and see `.jpg` files, not a zip.

The wet photos **must be new items** (none from 30 Oct) and must be taken with the **same plate and lighting**
as the 10 empty-plate photos; if the light changed, also take 3-5 new empty-plate photos and add them to
`background/` (the script picks the best-matching one for each photo).

### 2. The models (five files in one folder)

```bash
mkdir -p models
gh release download student-organik_gray_v2 -p weights.pt -O models/organik_gray_v2.pt
gh release download data-camera-day-2026-10-30 -p student_u1.pt -p student_u1_B.pt \
    -p teacher_v1_gray.joblib -p teacher_v2.joblib -D models
```

| File | Model |
|---|---|
| `organik_gray_v2.pt` | baseline student, before any loop |
| `student_u1.pt` | after the 30 Oct loop round (human labels + teacher soft labels) |
| `student_u1_B.pt` | variant B: same retraining, human labels only |
| `teacher_v1_gray.joblib` | teacher, public data only |
| `teacher_v2.joblib` | teacher after the loop (+ 21 bin crops) |

### 3. Contact sheets of the wet photos

```bash
python experiments/session2_2026-11-02/scripts/sheets.py session2/wet session2/sheets wet
```

Open `session2/sheets/wet_00.jpg`, `wet_01.jpg`, ... Each tile is numbered: that number is the photo's index.

### 4. Group the wet photos into items

In `experiments/session2_2026-11-02/scripts/items.py`, replace `"wet": []` with the groups, one list per
physical item, using the numbers from the sheets. Same rules as session 1 and the dry photos:

- consecutive shots of one item (turned over, moved) go in one group;
- **when unsure whether two shots are the same item, put them in one group** (merging is safe; splitting one
  item into two would count it twice);
- pieces of one thing (e.g. peels of one orange) go in one group;
- a photo with two items in it: keep it as one group if both are wet, and add a line to `NOTES`;
- every photo number must be used exactly once (the script stops and says which are missing or repeated).

Example: `"wet": [[0, 1, 2], [3, 4, 5], [6, 7], ...],`

Labels come from the folder (`wet/`). Agreed rule from session 1: dry items with food on them stay dry.

### 5. Run

```bash
bash experiments/session2_2026-11-02/scripts/run.sh session2 models
```

It writes `session2/items.csv`, crops every photo (`session2/crops/`), scores the five models
(`session2/preds/`) and writes `session2/scores_tables.md`. About 5-10 minutes on a CPU.

To check the setup before the wet photos are in, add `--dry_only` (scores the dry photos only; those numbers
are dry recall only and are not results).

### 6. Check the crops before using any number

Open `session2/crops/check/sheet_*.jpg`: every photo with its mask in red and the box in green.

- Each photo should have the item masked (partly masked transparent items are expected).
- The run prints `no item: [...]` for photos where nothing was found; those are left out of the scores and
  must be reported as detector misses.
- **Do not change the threshold (16) or any setting** to rescue a photo: session 2 is a test, and the settings
  are fixed from session 1. Report misses instead.

### 7. Results

Put into this folder (`experiments/session2_2026-11-02/`):

- `results/scores_tables.md` (from `session2/`)
- `results/items.csv`, `results/crops.csv` (from `session2/crops/`), `results/predictions/` (from `session2/preds/`)
- the updated `scripts/items.py` (with the wet groups)
- in this README: the item counts, the camera height, the no-item photos, and the answers to the four
  questions at the top, each with its p-value per item.

Upload the wet photos and the crops to release `data-session2-2026-11-02` (to be created) so the numbers can be
checked.

## What the tables contain

For each model, per photo and per item: accuracy with a 95% Wilson interval, balanced accuracy, precision /
recall / F1 per class, AUROC, ECE, the confusion matrix; the review policy at 0.8 (share sent to a person,
errors caught, accuracy of what the bin does alone); and exact McNemar tests between every pair of models, per
photo and per item. Use the per-item p-values in the paper (photos of one item are not independent).

## Not done here (on purpose)

- No second loop round and no retraining (that needs a separate loop pile and test set).
- No change to the review threshold (0.8), the detector settings or the repo's code.
- No tuning on session 2 photos.
