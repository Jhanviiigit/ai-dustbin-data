# Dataset D: background removed

Built from `dataset_C_v2` with rembg model `u2net`, images shrunk to max 512px, background gray 128. Masks are in `masks/`.

- Images: **2716** kept, **56** dropped (mask covers < 1% of the image)

## Dropped by source and class (check these are not lopsided)

| Source | Class | Dropped | Of | Rate |
|---|---|---|---|---|
| realwaste | dry | 1 | 846 | 0.1% |
| realwaste | wet | 17 | 846 | 2.0% |
| roboflow | dry | 10 | 540 | 1.9% |
| roboflow | wet | 28 | 540 | 5.2% |

## Kept images per split

| Split | Dry | Wet | Total |
|---|---|---|---|
| train | 963 | 939 | 1902 |
| val | 204 | 199 | 403 |
| test | 208 | 203 | 411 |

## Mask coverage (fraction of image kept), median by source and class

| Source | Dry | Wet |
|---|---|---|
| realwaste | 0.19 | 0.15 |
| roboflow | 0.28 | 0.29 |
