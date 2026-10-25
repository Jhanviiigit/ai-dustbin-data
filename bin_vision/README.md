# bin_vision: multi-item detection

The bin's plate tilts one way, so it may only tilt when every item on it belongs
in the same compartment. `multi_item.py` finds each item on the plate and decides:

| Decision | When | Bin should |
|---|---|---|
| `EMPTY` | nothing on the plate | wait |
| `TILT:DRY` / `TILT:WET` | every item is confidently the same class | tilt |
| `MIXED` | confident items disagree | not tilt; ask the user to put in one item at a time |
| `ASK_HUMAN` | some item is below the confidence threshold | not tilt; ask a human (human-in-the-loop) |

How it works:
1. **Background subtraction** against a photo of the empty plate (the camera
   never moves). No neural network, a few ms per frame.
2. The change mask is cleaned up and split into blobs. Fragments of one item
   that sit close together are merged.
3. Each blob is cropped with everything outside it grayed out (same look as
   dataset D) and classified with `WasteClassifier` from ai_dustbin_2.

## Run it

Needs the ai_dustbin_2 repo for `waste_classifier.py` and the model weights.

```bash
# Live webcam demo: clear the plate, press 'r' to capture the reference, 'q' to quit
python multi_item.py --webcam --classifier_dir ../../ai_dustbin_2

# One image pair
python multi_item.py --reference empty.jpg --frame scene.jpg --classifier_dir ../../ai_dustbin_2 --save out.jpg
```

## Synthetic test (until we have real bin photos)

`test_multi_item.py` pastes real waste cut-outs (dataset D masks, test split)
onto fake plates (3 colours, lighting gradient, sensor noise, exposure change,
soft shadows) and checks the decision.

```bash
python test_multi_item.py --clean ../../cleaned_dataset --masks ../../dataset_D/masks \
    --classifier_dir ../../ai_dustbin_2 --n 60
```

First results (60 scenes per scenario, threshold 0.6, the **old** ai_dustbin_2 model):

| Scenario | Count right | Correct | Asked human | Unsafe tilt |
|---|---|---|---|---|
| single dry | 97% | 93% | 3% | 2% |
| single wet | 100% | 93% | 2% | 5% |
| two dry | 80% | 92% | 8% | 0% |
| two wet | 92% | 88% | 12% | 0% |
| mixed, apart | 75% | 68% | 5% | 27% |
| **mixed, touching** | 92%* | 8% | 13% | **78%** |
| three, mixed | 60% | 78% | 3% | 18% |

\* For touching items the "right" count is 1 blob.

"Unsafe tilt" = tilted a mixed load, or tilted a single-class load the wrong way.

What this shows:
- **Separate items:** detection works. Most counting misses are one item broken
  into pieces (harmless when the pieces agree) or pale/transparent items close
  to the plate colour being partly missed.
- **Touching wet + dry items: not handled.** They merge into one blob that the
  classifier labels confidently as one class. Watershed splitting (`--split`)
  did not help (80% unsafe) and broke single items apart, so it is off by default.
  This needs an object detector (e.g. YOLOv8n) or a hardware/UX answer.
- **Classifier errors** cause much of the rest: the old model was trained on
  photos with backgrounds, not gray-background crops. A model trained on
  dataset D should match these crops better.
- Synthetic scenes are not real bin photos. Re-run on photos from the bin camera.
