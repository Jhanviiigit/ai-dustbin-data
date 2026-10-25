# bin_vision: item detection before classification

The bin takes **one item at a time**. `multi_item.py` runs on the Jetson before
the student model: it counts what is on the plate, warns when there is more
than one item, and only passes a single cropped item to the classifier.

| Decision | When | Bin should |
|---|---|---|
| `EMPTY` | nothing on the plate | wait |
| `MULTIPLE` | 2+ items (see below) | not tilt; show "one item at a time" |
| `TILT:DRY` / `TILT:WET` | one item, classified confidently | tilt |
| `ASK_HUMAN` | one item, below the confidence threshold | not tilt; ask a human (human-in-the-loop) |

How it works:
1. **Background subtraction** against a photo of the empty plate (the camera
   never moves). No neural network, a few ms per frame.
2. The change mask is cleaned up and split into blobs. Fragments of one item
   that sit close together are merged.
3. **One-item check** (`check_count`, no classifier): `MULTIPLE` if
   - there are 2+ blobs, or
   - the **drop counter** saw 2+ separate drops since the plate was last empty
     (catches items placed one after another, even if they end up touching), or
   - the single blob **looks like several items** (`ShapeRule`): bigger than one
     item, or deep notches on both sides of a join.
4. Exactly one item: it is cropped with everything outside it grayed out
   (same look as dataset D) and classified with `WasteClassifier`.

## Run it

Needs the ai_dustbin_2 repo for `waste_classifier.py` and the model weights.

```bash
# Live webcam demo: clear the plate, press 'r' to capture the reference, 'q' to quit
python multi_item.py --webcam --classifier_dir ../../ai_dustbin_2

# One image pair (no drop history, so only blob count + shape checks)
python multi_item.py --reference empty.jpg --frame scene.jpg --classifier_dir ../../ai_dustbin_2 --save out.jpg
```

## Synthetic tests (until we have real bin photos)

`synth.py` pastes real waste cut-outs (dataset D masks) onto fake plates
(3 colours, lighting gradient, sensor noise, exposure drift, soft shadows),
as still scenes or as frame sequences where items are dropped one by one.

### One-item warning (`test_counting.py`, no classifier)

```bash
python test_counting.py --clean ../../cleaned_v2 --masks ../../dataset_D_v2/masks --n 150 --out counting_results.md
```

Shape thresholds are tuned on scenes built from **val-split** items (at most 5%
false alarms on single items), then everything is measured on fresh scenes from
**test-split** items. 150 scenes per scenario:

| Method | False alarm (single item) | Two apart | Two touching, one by one | Two touching, together | Three apart |
|---|---|---|---|---|---|
| count only | 2% | 83% | 9% | 7% | 95% |
| + shape checks | 4% | 89% | 35% | 33% | 99% |
| + drop counter | 2% | 95% | 93% | 7% | 100% |
| **all three** | **4%** | **96%** | **94%** | **33%** | **100%** |

Numbers in the last four columns are warning recall: the share of multi-item
loads that got the `MULTIPLE` warning.

What this shows:
- **The drop counter is the big win.** Items placed one after another are
  caught 93-94% of the time even when they end up touching, with no extra false alarms.
- **Two touching items dropped together stay hard (33%).** Almost all of that
  comes from the size rule ("bigger than one item"), which depends on how big one
  item looks to the camera: re-tune `DEFAULT_RULE.max_area_frac` on real bin photos.
  Notch depth adds a little; colour differences did not help at all (dropped).
  This case needs a learned detector (e.g. YOLO): future work.
- **False alarms (2-4%)** are mostly one item broken into two blobs. Harmless
  for safety, but the user sees a wrong warning.
- Watershed splitting of blobs (`--split`) was tried earlier and did not help.

### Full pipeline with the classifier (`test_multi_item.py`)

```bash
python test_multi_item.py --clean ../../cleaned_v2 --masks ../../dataset_D_v2/masks \
    --classifier_dir ../../ai_dustbin_2 --n 60
```

This uses the older class-agreement policy (tilt when every item agrees), kept
to measure classifier errors. First results with the **old** ai_dustbin_2 model
(v1 data, 60 scenes per scenario):

| Scenario | Count right | Correct | Asked human | Unsafe tilt |
|---|---|---|---|---|
| single dry | 97% | 93% | 3% | 2% |
| single wet | 100% | 93% | 2% | 5% |
| mixed, apart | 75% | 68% | 5% | 27% |
| mixed, touching | 92%* | 8% | 13% | 78% |

\* For touching items the "right" count is 1 blob.

Re-run with the new student model once it is trained. Synthetic scenes are not
real bin photos: re-run everything on photos from the bin camera.
