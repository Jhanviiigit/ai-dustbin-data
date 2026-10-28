# Dataset D: background removed

Built from `cleaned_dataset` with rembg model `u2net`, images shrunk to max 512px, background gray 128. Masks are in `masks/`.

- Images: **9542** kept, **137** dropped (mask covers < 1% of the image)

## Dropped by source and class (check these are not lopsided)

| Source | Class | Dropped | Of | Rate |
|---|---|---|---|---|
| dataset2 | dry | 32 | 1989 | 1.6% |
| dataset2 | wet | 31 | 1631 | 1.9% |
| realwaste | dry | 3 | 3409 | 0.1% |
| realwaste | wet | 17 | 846 | 2.0% |
| roboflow | dry | 26 | 1264 | 2.1% |
| roboflow | wet | 28 | 540 | 5.2% |

## Kept images per split

| Split | Dry | Wet | Total |
|---|---|---|---|
| train | 4621 | 2059 | 6680 |
| val | 991 | 437 | 1428 |
| test | 989 | 445 | 1434 |

## Mask coverage (fraction of image kept), median by source and class

| Source | Dry | Wet |
|---|---|---|
| dataset2 | 0.23 | 0.39 |
| realwaste | 0.20 | 0.15 |
| roboflow | 0.28 | 0.29 |
