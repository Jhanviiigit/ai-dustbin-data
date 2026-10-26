"""
multi_item.py
-------------
Finds every item on the bin's plate, labels each one, and decides whether the
plate can tilt. The plate tilts one way, so a load may only tilt when every
item on it belongs in the same compartment (dry+dry or wet+wet).

  1. Background subtraction against a reference photo of the empty plate
     (the camera and plate never move, so this is cheap and needs no network).
  2. Clean up the change mask and split it into blobs. Fragments of one item
     that sit close together are merged.
  3. Label every item with the student model, one crop per item. Crops are the
     item's box + 10% padding, everything outside the item gray (128): the
     same look as dataset D. Items get their label from
       - their own blob, when items lie apart, or
       - the DropCounter, which crops and labels each item the moment it
         lands, while it is still alone. This keeps the labels of items that
         end up touching.
  4. assess() decides:
        EMPTY      nothing on the plate
        TILT:DRY   every item confidently dry  (dry, dry+dry, ...)
        TILT:WET   every item confidently wet  (wet, wet+wet, ...)
        MIXED      confident items disagree (dry+wet) -> don't tilt, ask the user to separate
        ASK_HUMAN  any item below the confidence threshold -> don't tilt, ask a human
        SEPARATE   one blob that looks like several touching items, with no drop
                   history to label them -> contents unknown, don't tilt, ask to separate
     Decision.composition is e.g. "dry+wet"; Decision.load_confidence is the
     lowest item confidence (a load is only as certain as its weakest item).

The student model is any object with classify_image(crop_bgr) returning
{"label": "dry" | "wet", "confidence": float, ...}; see make_classify().

Live demo (webcam; press 'r' with the plate empty to grab the reference, 'q' to quit):
  python multi_item.py --webcam --student_dir ../path/to/student --model_path student.pt

Single image pair:
  python multi_item.py --reference empty.jpg --frame scene.jpg --student_dir ... --model_path ...
"""

import argparse
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

GRAY = 128


@dataclass
class Item:
    box: tuple                     # x, y, w, h in frame pixels
    area_frac: float               # blob area / frame area
    mask: np.ndarray = field(repr=False)
    label: str = ""                # "dry" / "wet"
    confidence: float = 0.0
    crop: np.ndarray = field(default=None, repr=False)   # set by DropCounter: the crop classified at landing


@dataclass
class Decision:
    action: str                    # EMPTY, TILT:DRY, TILT:WET, MIXED, ASK_HUMAN, SEPARATE (check_count: SINGLE, MULTIPLE)
    items: list
    reason: str
    composition: str = ""          # e.g. "dry+wet"; "" when unknown
    load_confidence: float = 0.0   # lowest item confidence


class ItemDetector:
    """Background subtraction against an empty-plate reference."""

    def __init__(self, diff_threshold=28.0, min_area_frac=0.004, blur=5, work_width=480,
                 merge_frac=0.03, split_touching=False, split_solidity=0.8, split_peak=0.5):
        self.diff_threshold = diff_threshold
        self.min_area_frac = min_area_frac
        self.blur = blur
        self.work_width = work_width
        self.merge_frac = merge_frac            # fragments closer than this (fraction of width) are one item
        self.split_touching = split_touching
        self.split_solidity = split_solidity    # blobs less convex than this are split candidates
        self.split_peak = split_peak            # distance-transform peak level used as split seeds
        self.reference = None      # float32 Lab image at working size

    def _prep(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = self.work_width / w
        small = cv2.resize(frame_bgr, (self.work_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (self.blur, self.blur), 0)
        return cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)

    def set_reference(self, frames_bgr):
        """Average one or more frames of the empty plate."""
        if isinstance(frames_bgr, np.ndarray):
            frames_bgr = [frames_bgr]
        self.reference = np.mean([self._prep(f) for f in frames_bgr], axis=0)

    def update_reference(self, frame_bgr, rate=0.05):
        """Blend in a new empty-plate frame to follow slow lighting changes.
        Only call this when the distance sensor says the plate is empty."""
        self.reference = (1 - rate) * self.reference + rate * self._prep(frame_bgr)

    def change_mask(self, frame_bgr):
        """Binary mask (working size) of pixels that differ from the empty plate."""
        if self.reference is None:
            raise RuntimeError("Call set_reference() with an empty-plate frame first")
        lab = self._prep(frame_bgr)
        # Lightness-only changes are mostly shadows and exposure drift, so
        # they count for less than colour changes.
        lab = lab - (np.median(lab - self.reference, axis=(0, 1)))   # cancel global exposure shift
        d = np.abs(lab - self.reference)
        score = 0.5 * d[..., 0] + d[..., 1] + d[..., 2]
        mask = (score > self.diff_threshold).astype(np.uint8) * 255

        k = max(3, self.work_width // 160) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Fill holes (e.g. a transparent bottle's middle looks like the plate)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        return filled

    def _blobs(self, mask):
        """Label image of items: nearby fragments merged, pinched blobs split."""
        # Merge fragments of one item: group pixels that are within a few px of each other
        k = max(3, int(self.work_width * self.merge_frac)) | 1
        grouped = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, groups = cv2.connectedComponents(grouped, connectivity=8)
        labels = np.where(mask > 0, groups, 0).astype(np.int32)

        if not self.split_touching:
            return labels
        # Split blobs that look like two items meeting at a narrow neck
        next_label = n
        for i in range(1, n):
            blob = (labels == i).astype(np.uint8)
            area = int(blob.sum())
            if area < 400:
                continue
            contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull_area = cv2.contourArea(cv2.convexHull(np.vstack(contours)))
            if hull_area == 0 or area / hull_area > self.split_solidity:
                continue                                   # convex enough to be one item
            dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
            peaks = (dist > self.split_peak * dist.max()).astype(np.uint8)
            n_peaks, markers = cv2.connectedComponents(peaks)
            if n_peaks - 1 < 2:
                continue
            markers = markers + 1                          # 1 = background, 2.. = seeds
            markers[(blob > 0) & (peaks == 0)] = 0         # unknown region for watershed to fill
            ws = cv2.watershed(cv2.cvtColor(blob * 255, cv2.COLOR_GRAY2BGR), markers.astype(np.int32))
            for seed in range(2, n_peaks + 1):
                labels[(ws == seed) & (blob > 0)] = next_label
                next_label += 1
            labels[(ws == -1) & (blob > 0)] = 0            # watershed boundary pixels
        return labels

    def detect(self, frame_bgr):
        """Return one Item per blob, with boxes and masks at full frame size."""
        h, w = frame_bgr.shape[:2]
        labels_small = self._blobs(self.change_mask(frame_bgr))
        labels = cv2.resize(labels_small.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        items = []
        for i in np.unique(labels):
            if i == 0:
                continue
            m = labels == i
            area_frac = m.mean()
            if area_frac < self.min_area_frac:
                continue
            ys, xs = np.nonzero(m)
            box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
            items.append(Item(box=box, area_frac=area_frac, mask=m))
        return items


@dataclass
class ShapeRule:
    """When one blob probably holds more than one item. Any rule that fires counts.
    Defaults come from tuning on synthetic val-split scenes (test_counting.py)."""
    min_solidity: float = 0.0          # blob area / convex hull area below this -> several items
    max_area_frac: float = 1.0         # blob bigger than this share of the frame -> several items
    min_color_sep: float = float("inf")  # two colour clusters further apart than this -> several items
    min_color_minor: float = 0.2       # ... and the smaller cluster is at least this share of the blob
    min_notch2: float = float("inf")   # second-deepest notch (relative to blob size) above this -> several items


def blob_features(frame_bgr, item):
    """Shape and colour features of one blob, used by ShapeRule."""
    x, y, w, h = item.box
    sub = item.mask[y:y + h, x:x + w].astype(np.uint8)
    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    area = float(sub.sum())
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area > 0 else 1.0
    # Two items meeting leave notches on both sides of the join: the two
    # deepest convexity defects, relative to the blob's size
    notch1 = notch2 = 0.0
    try:
        defects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints=False))
        if defects is not None:
            depths = sorted((defects[:, 0, 3] / 256.0) / np.sqrt(area), reverse=True)
            notch1 = float(depths[0]) if len(depths) > 0 else 0.0
            notch2 = float(depths[1]) if len(depths) > 1 else 0.0
    except cv2.error:
        pass                                   # self-intersecting hull: leave notches at 0
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    elongation = max(rw, rh) / max(min(rw, rh), 1.0)

    # Two-cluster k-means on the blob's colours (Lab, lightness down-weighted
    # so shading on one object does not look like two objects)
    lab = cv2.cvtColor(frame_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab[sub > 0] * np.array([0.5, 1.0, 1.0], np.float32)
    if len(pixels) > 3000:
        pixels = pixels[np.random.default_rng(0).choice(len(pixels), 3000, replace=False)]
    color_sep, color_minor = 0.0, 0.0
    if len(pixels) >= 50:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()
        spread = np.sqrt(np.mean([pixels[labels == k].var(axis=0).sum() for k in (0, 1) if (labels == k).any()]))
        color_sep = float(np.linalg.norm(centers[0] - centers[1]) / max(spread, 1e-6))
        color_minor = float(min(labels.mean(), 1 - labels.mean()))
    return {"solidity": solidity, "area_frac": item.area_frac, "color_sep": color_sep, "color_minor": color_minor,
            "notch1": notch1, "notch2": notch2, "elongation": elongation}


def suspect_multiple(features, rule):
    """Reason string if the blob looks like several items, else None."""
    if features["solidity"] < rule.min_solidity:
        return f"blob not compact (solidity {features['solidity']:.2f})"
    if features["area_frac"] > rule.max_area_frac:
        return f"blob larger than one item ({features['area_frac']:.0%} of frame)"
    if features["notch2"] > rule.min_notch2:
        return f"notches on both sides of a join (depth {features['notch2']:.2f})"
    if features["color_sep"] > rule.min_color_sep and features["color_minor"] >= rule.min_color_minor:
        return f"two distinct colour regions (separation {features['color_sep']:.1f})"
    return None


# Tuned on synthetic val-split scenes (test_counting.py): <= 5% false alarms on single
# items. max_area_frac depends on how big one item looks to the camera, so
# re-tune it on real bin photos once the camera is mounted.
DEFAULT_RULE = ShapeRule(max_area_frac=0.09, min_notch2=0.5)


class DropCounter:
    """Counts separate drops onto the plate since it was last empty.

    Watches consecutive frames. When the picture stops changing for
    `settle_frames` frames, the plate has settled; if the foreground now
    covers a clearly new region compared with the last settled state, that is
    one more drop. Two items dropped one after another count 2 even if they
    end up touching. Given a classify function, each new item is cropped from
    its new region and labelled right away, while it is still alone; these
    labelled items, each with the crop it was labelled from (item.crop), are
    kept in drop_items. Call reset() after the plate tilts.
    """

    def __init__(self, detector, settle_frames=3, motion_level=20, motion_frac=0.004, min_new_frac=0.004):
        self.detector = detector
        self.settle_frames = settle_frames
        self.motion_level = motion_level      # grey-level change that counts as motion
        self.motion_frac = motion_frac        # share of pixels that must move to count as motion
        self.min_new_frac = min_new_frac      # new foreground needed to count a drop
        self.reset()

    def reset(self):
        self.drops = 0
        self.drop_items = []
        self.prev_gray = None
        self.still = 0
        self.settled_fg = None

    def update(self, frame_bgr, classify=None):
        small = cv2.resize(frame_bgr, (self.detector.work_width, int(round(frame_bgr.shape[0] * self.detector.work_width / frame_bgr.shape[1]))))
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.int16)
        moving = False
        if self.prev_gray is not None:
            moving = (np.abs(gray - self.prev_gray) > self.motion_level).mean() > self.motion_frac
        self.prev_gray = gray
        self.still = 0 if moving else self.still + 1
        if self.still == self.settle_frames:
            fg = self.detector.change_mask(frame_bgr) > 0
            if self.settled_fg is None:
                self.settled_fg = np.zeros_like(fg)
            grown = cv2.dilate(self.settled_fg.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            new = fg & ~grown
            if new.mean() > self.min_new_frac:
                self.drops += 1
                if classify is not None:
                    item = _item_from_mask(new, frame_bgr.shape[:2])
                    if item is not None:
                        item.label, item.confidence = classify(frame_bgr, item)
                        # Keep the crop the student just saw: later items may land on top,
                        # so re-cutting it from a later frame would no longer show this item alone.
                        item.crop = item_crop(frame_bgr, item)
                        self.drop_items.append(item)
            self.settled_fg = fg
        return self.drops


def _item_from_mask(small_mask, frame_hw):
    """Item at full frame size from a working-size boolean mask (largest part only)."""
    h, w = frame_hw
    m = cv2.resize(small_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = stats[i]
    return Item(box=(int(x), int(y), int(bw), int(bh)), area_frac=area / (w * h), mask=(labels == i))


def check_count(frame_bgr, items, rule=None, drops=None):
    """Count-only decision before any classification: EMPTY, SINGLE or MULTIPLE."""
    if not items:
        return Decision("EMPTY", items, "nothing on the plate")
    if len(items) > 1:
        return Decision("MULTIPLE", items, f"{len(items)} separate items")
    if drops is not None and drops > 1:
        return Decision("MULTIPLE", items, f"{drops} separate drops onto the plate")
    if rule is not None:
        reason = suspect_multiple(blob_features(frame_bgr, items[0]), rule)
        if reason:
            return Decision("MULTIPLE", items, reason)
    return Decision("SINGLE", items, "one item")


def item_crop(frame_bgr, item, pad=0.1):
    """Square-ish crop around the item with everything outside its mask grayed out."""
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = item.box
    p = int(max(bw, bh) * pad)
    x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(w, x + bw + p), min(h, y + bh + p)
    crop = frame_bgr[y0:y1, x0:x1].copy()
    crop[~item.mask[y0:y1, x0:x1]] = GRAY
    return crop


def make_classify(classifier):
    """classify(frame, item) -> (label, confidence) using the student model.

    The student is any object with classify_image(crop_bgr) returning
    {"label": "dry" | "wet", "confidence": float, ...}; "dry waste" / "wet waste"
    (the dataset folder names) are accepted too.
    """
    def classify(frame_bgr, item):
        result = classifier.classify_image(item_crop(frame_bgr, item))
        label = str(result["label"]).lower().replace(" waste", "").strip()
        if label not in ("dry", "wet"):
            raise ValueError(f"Student returned label {result['label']!r}; expected 'dry' or 'wet'")
        return label, float(result["confidence"])
    return classify


def classify_items(frame_bgr, items, classify):
    """Fill in label/confidence for each item."""
    for item in items:
        item.label, item.confidence = classify(frame_bgr, item)
    return items


def decide(items, threshold=0.6):
    """Agreement rule over labelled items: TILT, MIXED or ASK_HUMAN."""
    if not items:
        return Decision("EMPTY", items, "nothing on the plate")
    composition = "+".join(sorted(it.label for it in items))
    load_conf = min(it.confidence for it in items)
    unsure = [it for it in items if it.confidence < threshold]
    labels = {it.label for it in items if it.confidence >= threshold}
    if len(labels) > 1:
        return Decision("MIXED", items, f"{composition}: wet and dry together, please separate", composition, load_conf)
    if unsure:
        return Decision("ASK_HUMAN", items, f"{composition}: {len(unsure)} of {len(items)} item(s) below {threshold:.0%} confidence",
                        composition, load_conf)
    label = labels.pop()
    return Decision(f"TILT:{label.upper()}", items, f"{composition}: all {label}", composition, load_conf)


def assess(frame_bgr, items, classify, threshold=0.6, rule=None, drop_items=None):
    """Full decision for the current plate: label every item, then decide().

    items       blobs from ItemDetector.detect(frame)
    drop_items  items labelled by DropCounter as they landed (optional)
    """
    if not items:
        return Decision("EMPTY", items, "nothing on the plate")
    drop_items = drop_items or []
    if len(drop_items) > len(items):
        # More drops than blobs: some items merged after landing. Their labels
        # from the moment they landed are the only reliable ones.
        return decide(drop_items, threshold)
    if len(items) == 1 and rule is not None:
        reason = suspect_multiple(blob_features(frame_bgr, items[0]), rule)
        if reason:
            return Decision("SEPARATE", items, f"probably several touching items ({reason}); contents unknown")
    return decide(classify_items(frame_bgr, items, classify), threshold)


def draw(frame_bgr, decision):
    out = frame_bgr.copy()
    colours = {"dry": (255, 160, 0), "wet": (0, 170, 0)}
    for it in decision.items:
        x, y, bw, bh = it.box
        c = colours.get(it.label, (0, 0, 255))
        cv2.rectangle(out, (x, y), (x + bw, y + bh), c, 2)
        cv2.putText(out, f"{it.label} {it.confidence:.0%}", (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    banner = {"MIXED": (0, 0, 255), "MULTIPLE": (0, 0, 255), "SEPARATE": (0, 0, 255),
              "ASK_HUMAN": (0, 140, 255)}.get(decision.action, (40, 40, 40))
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), banner, -1)
    cv2.putText(out, f"{decision.action}: {decision.reason}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def load_student(student_dir, model_path, module="classifier"):
    """Import the student's WasteClassifier from <student_dir>/<module>.py (or package)."""
    sys.path.insert(0, os.path.abspath(student_dir))
    WasteClassifier = getattr(__import__(module), "WasteClassifier")
    return WasteClassifier(model_path=model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-item detection on the bin plate")
    parser.add_argument("--student_dir",    default=None, help="Folder containing the student model's code (leave out for count-only mode)")
    parser.add_argument("--student_module", default="classifier", help="Module in student_dir that defines WasteClassifier")
    parser.add_argument("--model_path",     default=None, help="Student weights file")
    parser.add_argument("--threshold",      type=float, default=0.6)
    parser.add_argument("--webcam",         action="store_true")
    parser.add_argument("--camera",         type=int, default=0)
    parser.add_argument("--reference",      help="Photo of the empty plate")
    parser.add_argument("--frame",          help="Photo with items on the plate")
    parser.add_argument("--save",           help="Where to save the annotated frame (image mode)")
    args = parser.parse_args()

    classify = None
    if args.student_dir:
        if not args.model_path:
            parser.error("--model_path is needed with --student_dir")
        classify = make_classify(load_student(args.student_dir, args.model_path, args.student_module))
    else:
        print("No student model given: count-only mode (EMPTY / SINGLE / MULTIPLE).")

    def evaluate(frame, items, drops=None, drop_items=None):
        if classify is None:
            return check_count(frame, items, DEFAULT_RULE, drops)
        return assess(frame, items, classify, args.threshold, DEFAULT_RULE, drop_items)

    detector = ItemDetector()
    counter = DropCounter(detector)

    if args.webcam:
        cap = cv2.VideoCapture(args.camera)
        print("Clear the plate and press 'r' to capture the reference. 'q' quits.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if detector.reference is None:
                shown = frame.copy()
                cv2.putText(shown, "Empty the plate, then press 'r'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                items = detector.detect(frame)
                counter.update(frame, classify)
                if not items:
                    counter.reset()                      # plate cleared: start counting drops again
                shown = draw(frame, evaluate(frame, items, counter.drops, counter.drop_items))
            cv2.imshow("multi-item", shown)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                detector.set_reference([cap.read()[1] for _ in range(10)])
                print("Reference captured.")
            elif key == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        if not (args.reference and args.frame):
            parser.error("Give --webcam, or both --reference and --frame")
        detector.set_reference(cv2.imread(args.reference))
        frame = cv2.imread(args.frame)
        items = detector.detect(frame)
        decision = evaluate(frame, items)          # a single photo has no drop history
        print(f"{decision.action}: {decision.reason}" +
              (f"  (load confidence {decision.load_confidence:.0%})" if decision.composition else ""))
        for it in decision.items:
            print(f"  {it.label:3s} {it.confidence:.0%}  box={it.box}  area={it.area_frac:.1%}")
        if args.save:
            cv2.imwrite(args.save, draw(frame, decision))
            print(f"Saved {args.save}")
