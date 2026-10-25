# Dataset cleaning report

- Sources: dataset2, realwaste, roboflow
- Input images: **9725** (RealWaste 'Miscellaneous Trash' left out: 495)
- Kept: **9679**  |  Dropped: **46**
- Duplicate groups: 174 (of which 58 had copies in both the old ai_dustbin_2 train and val folders)
- Near-duplicate threshold: pHash Hamming distance <= 4 / 64
- Near-duplicates: all kept, each group confined to one split
- Duplicate groups spanning sources: {'dataset2 + roboflow': 58, 'dataset2 + realwaste': 2}
- Duplicate groups spanning more than one new split: 0
- Split ratios: [0.7, 0.15, 0.15], seed 42

## Dropped, by reason

| Reason | Count |
|---|---|
| exact duplicate | 44 |
| near duplicate with conflicting labels | 2 |

## Kept images per split

| Split | Dry (dataset2) | Dry (realwaste) | Dry (roboflow) | Wet (dataset2) | Wet (realwaste) | Wet (roboflow) | Total | Wet share |
|---|---|---|---|---|---|---|---|---|
| train | 1396 | 2384 | 885 | 1143 | 592 | 378 | 6778 | 31.2% |
| val | 297 | 511 | 191 | 245 | 127 | 81 | 1452 | 31.2% |
| test | 296 | 514 | 188 | 243 | 127 | 81 | 1449 | 31.1% |

## Wet share within each source (kept images)

| Source | Dry | Wet | Wet share |
|---|---|---|---|
| dataset2 | 1989 | 1631 | 45.1% |
| realwaste | 3409 | 846 | 19.9% |
| roboflow | 1264 | 540 | 29.9% |

## Class weights for training (as train_classifier.py computes them)

- dry: 0.726
- wet: 1.604

Every split has the same source mix, so per-source results can be reported from `manifest.csv`.
