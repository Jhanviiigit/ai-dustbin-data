# update: the human-in-the-loop update pipeline

Turns human feedback from the bin into a better student, safely. Built in phases:
**phase 1 (the shared pool)**, **phase 2 (feedback → pool)** and **phase 3 (`run_update.py`:
teacher, student, gate, deploy, `models.csv`)** are done. Phase 3 calls
`update_student.py --pool` (`student/README.md`, "One update of the student") and, for the teacher's
soft labels, `teacher/pool_soft_labels.py`.

| File | What it does |
|---|---|
| `pool.py` | The pool: read, add (with the agreed rules enforced), save, check; seed from `dataset_organik_v2` or `dataset_bin` |
| `baseline.py` | Ship the first student once (gate as a first release, deploy, first `models.csv` row) |
| `ledger.py` | Which feedback rows are in the pool, when the last update was, how many updates (`pool/ledger.json`) |
| `collect.py` | New feedback rows, filtered by the student's `load_feedback()` rules |
| `split.py` | Where each new group goes: by group and hash, with near-copy (leak) checks |
| `trigger.py` | Whether there is enough new training data for an update |
| `augment.py` | Balances `train` with capped, reproducible augmented copies |
| `ingest.py` | Ties them together: dry run, or `--commit` |
| `metrics.py` | Errors, accuracy, balanced accuracy, per-class P/R/F1, Wilson intervals, paired comparison + McNemar |
| `run_update.py` | One full update: ingest → teacher → student → gate (`student/promote.py`) → deploy → `models.csv` |
| `make_sample_pool.py` | A sample pool for building readers against |
| `test_pool.py`, `test_ingest.py`, `test_run_update.py` | Tests for every rule |

## Feedback → pool (`ingest.py`)

```bash
python update/pool.py seed --pool pool --organik dataset_organik_v2     # once
python update/ingest.py --pool pool --feedback feedback                  # dry run: plan + trigger, writes nothing
python update/ingest.py --pool pool --feedback feedback --commit         # commit if the trigger says go
python update/ingest.py --pool pool --public_index public_phash.csv      # public images near-copying a test image
```

**Where each new group goes** (feedback `load_id`; near-copies within a batch merged):

1. group already in the pool → its split (splits are permanent)
2. any image is a near-copy (pHash ≤ 4) of **any** pool image → the whole group joins that
   image's group and split. Blocks leaks both ways; an audit that lands in train/val this way
   is logged as `audit_leak`
3. audit → `test_audit`
4. low confidence: `bucket(group, 5) == 0` → `test_hard`; `bucket(group, 100) < 15` → `val`; else `train`

**Trigger** (counted after the split, in groups, train only): new train groups
≥ max(15, 15% of train groups) **and** ≥ 3 per class; **or** ≥ 14 days since the last update
**and** ≥ 10 new train groups. If not triggered, nothing is written and the rows stay pending.

**On commit:** every row goes to its split; `train` is balanced with augmented copies (minority
class, up to balance, ≤ 2 per original, flip / ±15° gray-filled rotation / ±20% brightness,
contrast, saturation, seeded from the copy's `crop_id`); the ledger marks the rows processed;
`pool/updates/update_N.json` records where every row went and why.

**Public data:** `--public_index` (CSV of `path, phash`) writes `pool/public_exclude.csv`, the
public images that near-copy a pool test image. The teacher leaves these out of training.

## One full update (`run_update.py`)

```bash
python update/run_update.py --pool pool --feedback feedback --current student/releases/student_v1/weights.pt  # first time
python update/run_update.py --pool pool --feedback feedback      # later: the current one is deploy/current
python update/run_update.py --pool pool                          # carry on after the teacher ran
```

| Step | What happens |
|---|---|
| 1. ingest | `ingest.py --commit`, only when the last update has finished. Trigger says wait → exit 2 |
| 2. teacher | Waits (exit 3) for `runs/teacher_v(N+1)/soft_labels.csv` (`python teacher/pool_soft_labels.py --pool pool --version teacher_v(N+1) --gray dataset_D_v2_balanced --out <that path>`, e.g. on Colab; the teacher is fitted on the public datasets plus the human-checked feedback crops and **never on Organik**; `teacher_v1` is the public-data teacher), then checks it: a row for every `train` `crop_id`, none for val/test, probabilities in [0, 1] |
| 3. student | `update_student.py --current … --version student_uN --pool … --soft_labels … --out runs/student_uN` |
| 4. gate | `promote.py --pool`: ≤ 2 more errors than the current **and** the best shipped version on `test_fixed`, and on `test_audit` from 30 images; `test_hard` reported only |
| 5. deploy | accepted → `deploy/current/{weights.pt, info.json}` (version, threshold, temperature); the old one → `deploy/previous` |
| 6. record | a row in `models.csv`, deployed **or** rejected, with test errors, accuracies, the paired margin and the reasons |

Run it again at any point: it carries on from the stage stored in `pool/updates/update_N.json`
(`"run"`). Update N's names are `teacher_v(N+1)` and `student_uN`; a rejected name is never reused.
Threshold for the bin: the student's `update.json` `"threshold"` (picked on val); 0.8 until
the student writes one (`models.csv` says which).

## The pool

Only the pipeline writes it (through `pool.py`); the teacher and student only read it.

```
pool/
  train/      {dry waste,wet waste}/<crop_id>.jpg   public (Organik) + feedback + augmented copies
  val/        ...                                   picks tau, the threshold, calibration
  test_fixed/ ...                                   dataset_bin/test, never changes; the gate's score
  test_audit/ ...                                   audit labels: accuracy of automatic decisions
  test_hard/  ...                                   some low-confidence groups; reported, never gated
  manifest.csv
```

`manifest.csv` columns (agreed format):

| Column | Values |
|---|---|
| `crop_id` | unique; the file name without `.jpg` |
| `path` | relative to `pool/`, forward slashes, e.g. `train/dry waste/abc.jpg` |
| `split` | `train`, `val`, `test_fixed`, `test_audit`, `test_hard` |
| `label` | `dry`, `wet` (folders are `dry waste` / `wet waste`) |
| `group_id` | public: `organik_g<group>` (Sian's similarity groups); capture: `item_id`; feedback: `load_id`; augmented: same as its original |
| `origin` | `public` (a starting dataset: Organik), `capture` (bin-camera captures), `feedback`, `augmented` |
| `parent_crop_id` | augmented copies: the original's `crop_id`; otherwise empty |
| `trigger` | feedback: `low_confidence` or `audit`; otherwise empty |
| `from_drop` | feedback: `1` (crop saved when the item landed) or `0`; otherwise empty |
| `phash` | 64-bit perceptual hash, 16 hex digits |
| `added_in` | the update (or `seed`) that added the row |
| `student_conf` | feedback: the student's confidence when it was labelled; otherwise empty |

Rules enforced by `pool.py`: unique `crop_id`s; only the agreed values; **a group's split
never changes**; **test splits hold real images only**; an augmented copy points to an
existing original in train, in the same group.

**Readers** (teacher, student): train on `train` only (plus public data); the teacher also leaves out the
Organik rows (`origin` `public` with `group_id` `organik_g<N>`, and their augmented copies); use `val` for
settings; score `test_fixed`, `test_audit` and `test_hard` separately; never train on
`val` or any test split. Map `label` `dry` -> 0, `wet` -> 1. Use `group_id` for grouped folds.

```bash
python update/pool.py seed  --pool pool --organik dataset_organik_v2   # start the real pool
python update/pool.py seed  --pool pool --dataset_bin dataset_bin     # (bin-camera captures, if any)
python update/pool.py check --pool pool                              # rule violations + counts
python update/test_pool.py                                           # unit tests
```

## The real pool and the baseline

The starting data is Sian's Organik split (release `data-organik-v2`; use the released zip, the
split cannot be rebuilt identically on another machine):

| Organik split | Pool split | Images (dry / wet) |
|---|---|---|
| train | `train` | 502 (170 / 332) |
| val | `val` | 72 (36 / 36) |
| test | `test_fixed` | 72 (36 / 36): fixed from now on, the gate's score |

82 groups (`organik_g<group>`), none in two splits; no val/test image is a pHash near-copy of a
train image. Every row is `origin = public`, `added_in = seed`, so bin numbers and Organik
numbers can always be told apart. At the first update, `train` is balanced with augmented dry
copies (≤ 2 per original), as for any update.

```bash
python update/pool.py seed --pool pool --organik dataset_organik_v2
python update/baseline.py --pool pool --weights student/runs/organik/weights.pt
```

`baseline.py` ships `organik_gray_v2` (release `student-organik_gray_v2`, trained on the Organik
train split only) once: the gate as a first release (on `test_fixed`: 7 errors, 90.3%), then
`student/releases/organik_gray_v2/`, `deploy/current/` with threshold 0.8, and the first
`models.csv` row (`update = seed`). It refuses to run a second time.

`test_fixed` is Organik phone photos, not the bin camera: it guards against a student getting
worse. The accuracy on the bin is `test_audit` (the 1-in-10 audits), gated from 30 images.

**Threshold:** 0.8 (`THRESHOLD_FALLBACK`, also in `bin_vision/hitl.py`) until the student's
val rule can be met (Wilson lower bound of precision ≥ 95% per class needs ≥ 73 val images per
class; there are 36). On the Organik test set 0.8 caught all 7 errors and asked a person about
15% of items; 0.6 caught 2.

**Read these numbers with care.** The 0.8 was read off the same 72 images that are now
`test_fixed`, so "7 of 7 caught, 15% asked" is not an independent test result: it is where the
threshold came from. Five of the 7 errors are one item (a green plastic straw), and 72 images is
too few to report. The headline accuracy of the student is the 5-fold cross-validation result
in [`student/README.md`](../student/README.md) (96.75%, std 1.4); the 90.3% on `test_fixed` only
shows that the pipeline runs. Treat 0.8 as a starting value and replace it with a threshold
picked on val once the val rule can be met, or on bin-camera audits.

## Tested end to end (2026-10-29), on real data

There is no bin yet, so the bin was replayed with real photos. Every component is the real one:
`pool.py`, `baseline.py`, `ingest.py`, `teacher/pool_soft_labels.py`, `update_student.py --pool`,
`promote.py`, `run_update.py`; no stand-ins. The only stand-ins are the *inputs*: the "bin feedback"
is real Organik photos that were held back, and the label a person would give is the dataset's label.

**The teacher** (`teacher/pool_soft_labels.py`) is fitted on the public datasets (`dataset_D_v2_balanced`,
optionally `dataset_C_v2`) plus the human-checked feedback crops and their augmented copies. It never
trains on Organik or on `test_fixed`. C and the temperature are picked on the public val splits plus the
pool's non-Organik val rows. Organik train images get the full teacher's soft labels (it never saw them);
feedback crops get out-of-fold labels. The run below was made with the earlier version of the script,
which fitted on the pool's Organik rows; it has not been repeated with the current one yet.

| Step | What happened |
|---|---|
| seed | `pool.py seed --organik` on `dataset_organik_v2` minus 24 held-back train groups (124 images): train 378, val 72, `test_fixed` 72 |
| baseline | `train.py --pool` (from ImageNet, 15 epochs, val 95.8%), calibrated (T 1.0075), shipped by `baseline.py`: 13 errors on `test_fixed` (81.9%) |
| the bin | the deployed student labelled the held-back photos; its review policy (confidence < 0.8, plus 1 in 10 audits) sent 33 of 124 to a person (23 unsure, 10 audits); 13 of the 33 it had wrong |
| ingest | trigger said wait (6 of 15 new train groups, 1 wet group of the 3 needed), so `--force`; 20 train, 5 val, 7 `test_audit`, 1 `test_hard` |
| teacher | `pool_soft_labels.py` (earlier version, fitted on the pool's Organik rows): 544 train soft labels (378 Organik, 146 augmented, 20 feedback); val 96.1% |
| student | `update_student.py --pool`: 10 epochs from the current student, T 0.8935 fitted on 77 val images, threshold null (the val rule cannot be met with 77 images) |
| gate | accepted: 15 errors vs 13 on `test_fixed` (+2, the limit; McNemar p 0.5), `test_audit` 7 images (reported only) |
| deploy | `deploy/current` = `student_u1`, threshold 0.8 (fallback), `models.csv` row `update_1` |

What this shows: the parts work together, from feedback to a deployed student. What it does **not**
show is that an update helps: the new student was not better (the gate allows up to 2 more errors),
the batch was 20 feedback crops, and the first student was trained on a deliberately small slice.
Run it again on real bin feedback.

## Sample pool

Release [`sample-pool-v0`](https://github.com/Jhanviiigit/ai-dustbin-data/releases/tag/sample-pool-v0)
holds a pool built from **synthetic** data (capture-tool and human-in-the-loop tests), with
every split and origin filled in. Use it to build and test readers; don't train real models on it.
