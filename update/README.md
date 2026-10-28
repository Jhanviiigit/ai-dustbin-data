# update: the human-in-the-loop update pipeline

Turns human feedback from the bin into a better student, safely. Built in phases:
**phase 1 (the shared pool)** and **phase 2 (feedback → pool)** are done; phase 3
(teacher, student, gate, deploy, `models.csv` in `run_update.py`) is next.

| File | What it does |
|---|---|
| `pool.py` | The pool: read, add (with the agreed rules enforced), save, check; seed from `dataset_bin` |
| `ledger.py` | Which feedback rows are in the pool, when the last update was, how many updates (`pool/ledger.json`) |
| `collect.py` | New feedback rows, filtered by the student's `load_feedback()` rules |
| `split.py` | Where each new group goes: by group and hash, with near-copy (leak) checks |
| `trigger.py` | Whether there is enough new training data for an update |
| `augment.py` | Balances `train` with capped, reproducible augmented copies |
| `ingest.py` | Ties them together: dry run, or `--commit` |
| `make_sample_pool.py` | A sample pool for building readers against |
| `test_pool.py`, `test_ingest.py` | Tests for every rule |

## Feedback → pool (`ingest.py`)

```bash
python update/pool.py seed --pool pool --dataset_bin dataset_bin        # once
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

## The pool

Only the pipeline writes it (through `pool.py`); the teacher and student only read it.

```
pool/
  train/      {dry waste,wet waste}/<crop_id>.jpg   capture + feedback + augmented copies
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
| `group_id` | capture: `item_id`; feedback: `load_id`; augmented: same as its original |
| `origin` | `capture`, `feedback`, `augmented` |
| `parent_crop_id` | augmented copies: the original's `crop_id`; otherwise empty |
| `trigger` | feedback: `low_confidence` or `audit`; otherwise empty |
| `from_drop` | feedback: `1` (crop saved when the item landed) or `0`; otherwise empty |
| `phash` | 64-bit perceptual hash, 16 hex digits |
| `added_in` | the update (or `seed`) that added the row |
| `student_conf` | feedback: the student's confidence when it was labelled; otherwise empty |

Rules enforced by `pool.py`: unique `crop_id`s; only the agreed values; **a group's split
never changes**; **test splits hold real images only**; an augmented copy points to an
existing original in train, in the same group.

**Readers** (teacher, student): train on `train` only (plus public data); use `val` for
settings; score `test_fixed`, `test_audit` and `test_hard` separately; never train on
`val` or any test split. Map `label` `dry` -> 0, `wet` -> 1. Use `group_id` for grouped folds.

```bash
python update/pool.py seed  --pool pool --dataset_bin dataset_bin   # start a pool
python update/pool.py check --pool pool                              # rule violations + counts
python update/test_pool.py                                           # unit tests
```

## Sample pool

Release [`sample-pool-v0`](https://github.com/Jhanviiigit/ai-dustbin-data/releases/tag/sample-pool-v0)
holds a pool built from **synthetic** data (capture-tool and human-in-the-loop tests), with
every split and origin filled in. Use it to build and test readers; don't train real models on it.
