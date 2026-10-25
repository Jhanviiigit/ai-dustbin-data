# Dataset D: background removed

Built from `cleaned_v2` with rembg model `u2net`, images shrunk to max 512px, background gray 128. Masks are in `masks/`.

- Images: **6015** kept, **74** dropped (mask covers < 1% of the image)

## Dropped by source and class (check these are not lopsided)

| Source | Class | Dropped | Of | Rate |
|---|---|---|---|---|
| realwaste | dry | 3 | 3409 | 0.1% |
| realwaste | wet | 17 | 846 | 2.0% |
| roboflow | dry | 26 | 1294 | 2.0% |
| roboflow | wet | 28 | 540 | 5.2% |

## Kept images per split

| Split | Dry | Wet | Total |
|---|---|---|---|
| train | 3275 | 939 | 4214 |
| val | 699 | 199 | 898 |
| test | 700 | 203 | 903 |

## Mask coverage (fraction of image kept), median by source and class

| Source | Dry | Wet |
|---|---|---|
| realwaste | 0.20 | 0.15 |
| roboflow | 0.28 | 0.29 |
