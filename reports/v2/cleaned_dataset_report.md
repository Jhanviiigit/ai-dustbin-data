# Dataset cleaning report

- Sources: realwaste, roboflow (left out: dataset2)
- Input images: **6091** (RealWaste 'Miscellaneous Trash' left out: 495)
- Kept: **6089**  |  Dropped: **2**
- Duplicate groups: 30 (of which 10 had copies in both the old ai_dustbin_2 train and val folders)
- Near-duplicate threshold: pHash Hamming distance <= 4 / 64
- Near-duplicates: all kept, each group confined to one split
- Duplicate groups spanning sources: none
- Duplicate groups spanning more than one new split: 0
- Split ratios: [0.7, 0.15, 0.15], seed 42

## Dropped, by reason

| Reason | Count |
|---|---|
| near duplicate with conflicting labels | 2 |

## Kept images per split

| Split | Dry (realwaste) | Dry (roboflow) | Wet (realwaste) | Wet (roboflow) | Total | Wet share |
|---|---|---|---|---|---|---|
| train | 2387 | 906 | 592 | 378 | 4263 | 22.8% |
| val | 511 | 194 | 127 | 81 | 913 | 22.8% |
| test | 511 | 194 | 127 | 81 | 913 | 22.8% |

## Wet share within each source (kept images)

| Source | Dry | Wet | Wet share |
|---|---|---|---|
| realwaste | 3409 | 846 | 19.9% |
| roboflow | 1294 | 540 | 29.4% |

## Class weights for training (as train_classifier.py computes them)

- dry: 0.647
- wet: 2.197

Every split has the same source mix, so per-source results can be reported from `manifest.csv`.
