# update: the human-in-the-loop update pipeline

Turns human feedback from the bin into a better student, safely. Built in phases;
this folder currently holds **phase 1: the shared pool**.

| File | What it does | Status |
|---|---|---|
| `pool.py` | The pool: read, add (with the agreed rules enforced), save, check; seed from `dataset_bin` | done |
| `test_pool.py` | Unit tests for the pool's rules | done |
| `make_sample_pool.py` | A sample pool from `dataset_bin` + a feedback folder, for building readers against | done |
| `collect.py`, `split.py`, `augment.py`, `run_update.py` | Ledger, group/hash split + leak checks, trigger, augmentation, full cycle | phase 2-3 |

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
