# bin_vision: item detection before classification

`multi_item.py` runs on the Jetson before the student model. It finds every
item on the plate, sends **one crop per item** to the student, and lets the
plate tilt only when every item belongs in the same compartment.

| Decision | When | Bin should |
|---|---|---|
| `EMPTY` | nothing on the plate | wait |
| `TILT:DRY` / `TILT:WET` | every item confidently the same class (dry, dry+dry, wet+wet, ...) | tilt |
| `MIXED` | confident items disagree (dry+wet) | not tilt; ask the user to separate |
| `ASK_HUMAN` | any item below the confidence threshold | not tilt; ask a human (human-in-the-loop) |
| `SEPARATE` | one blob that looks like several touching items, with no drop history to label them | not tilt; ask the user to separate |

Each decision also carries `composition` (e.g. `"dry+wet"`) and
`load_confidence` (the lowest item confidence).

How it works:
1. **Background subtraction** against a photo of the empty plate (the camera
   never moves). No neural network, a few ms per frame.
2. The change mask is cleaned up and split into blobs. Fragments of one item
   that sit close together are merged.
3. **Each item gets a label** from the student, one crop per item (`item_crop`:
   box + 10% padding, everything outside the item gray 128, same look as dataset D):
   - items lying apart: one crop per blob;
   - the **drop counter** crops and labels each item **the moment it lands**,
     while it is still alone, so items that end up touching keep their own labels.
4. `assess()` applies the table above. A single blob that looks like several
   items (`ShapeRule`: bigger than one item, or deep notches on both sides of
   a join) with no drop history is `SEPARATE`: its contents can't be labelled.

## The student model interface

The detector works with any object that has:

```python
result = student.classify_image(crop_bgr)   # numpy uint8 H×W×3, BGR, one item, gray 128 around it
# {"label": "dry" | "wet", "confidence": float (calibrated), "dry_prob": ..., "wet_prob": ..., "inference_ms": ...}
```

`load_student()` imports `WasteClassifier` from `<student_dir>/<student_module>.py`.

## Run it

```bash
# Live webcam demo: clear the plate, press 'r' to capture the reference, 'q' to quit
python multi_item.py --webcam --student_dir ../../student --model_path ../../student/student.pt

# Without a student model: count-only mode (EMPTY / SINGLE / MULTIPLE)
python multi_item.py --webcam

# One image pair (no drop history, so blob count + shape checks only)
python multi_item.py --reference empty.jpg --frame scene.jpg --student_dir ... --model_path ... --save out.jpg
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

### Same compartment or not (`test_composition.py`)

```bash
python test_composition.py --clean ../../cleaned_v2 --masks ../../dataset_D_v2/masks --n 80 --out composition_results.md
```

Runs the full detector and `assess()` on frame sequences. The student is
replaced by a **stand-in** that reads each crop's true label from the scene
(a merged dry+wet blob gets its majority label, confidently: the worst case).
Run once with a perfect stand-in, and once that is wrong 5% of the time and
unsure 5% of the time. 80 scenes per load, test-split items:

| Load | Correct | Stopped (safe) | Unsafe tilt | Correct, 5% wrong + 5% unsure | Unsafe, 5% wrong + 5% unsure |
|---|---|---|---|---|---|
| dry | 91% | 9% | 0% | 81% | 6% |
| wet | 95% | 5% | 0% | 86% | 4% |
| dry+dry apart | 100% | 0% | 0% | 84% | 0% |
| wet+wet apart | 100% | 0% | 0% | 80% | 0% |
| dry+wet apart | 91% | 1% | 8% | 71% | 18% |
| dry+dry touching, one by one | 99% | 1% | 0% | 69% | 1% |
| dry+wet touching, one by one | 95% | 1% | 4% | 81% | 10% |
| dry+dry touching, together | 72% | 28% | 0% | 66% | 2% |
| **dry+wet touching, together** | 4% | 16% | **80%** | 2% | **76%** |
| three, mixed, apart | 95% | 0% | 5% | 86% | 10% |

"Correct" = tilted the right way, or `MIXED` for a dry+wet load. "Stopped" =
did not tilt when it could have (safe, but the user has to act). "Unsafe" =
tilted a dry+wet load, or tilted the wrong way.

What this shows:
- **dry+dry, wet+wet and dry+wet are told apart** when the items lie apart or
  arrive one by one: 91-100% correct with a perfect classifier.
- **Classifier mistakes add up across items.** With a 5%-wrong student, unsafe
  tilts on mixed loads rise to 10-18%: every item is another chance to be wrong.
  Calibrated confidence and the human-in-the-loop threshold matter more here.
- **dry+wet dropped together and touching is still unsafe (~80%)**: one blob,
  no drop history, and the shape rule catches only ~20%. A learned detector
  (future work) or the "one item at a time" prompt is the answer.
- The 8% unsafe on dry+wet apart (perfect stand-in) is when the two blobs were
  detected as one and the drop counter also missed the second drop.

Synthetic scenes are not real bin photos: re-run everything on photos from
the bin camera, and with the real student once it is trained.
