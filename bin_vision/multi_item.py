"""
multi_item.py
-------------
Finds every item on the bin's plate and decides whether the plate can tilt.

The plate tilts one way, so a load is only safe to tilt when every item on it
belongs in the same compartment. Pipeline:

  1. Background subtraction against a reference photo of the empty plate
     (the camera and plate never move, so this is cheap and needs no network).
  2. Clean up the change mask and split it into connected blobs = items.
     Fragments of one item that sit close together are merged. Optional
     (split_touching=True): blobs with a narrow "neck" are split with a
     watershed. Off by default: on synthetic scenes it did not reduce unsafe
     tilts for touching wet+dry items and it broke single items apart.
  3. Crop each item, gray out everything outside its mask (same look as the
     background-removed training set, dataset D), and classify it.
  4. Decide:
        EMPTY      nothing on the plate
        TILT:DRY   every item is confidently dry
        TILT:WET   every item is confidently wet
        MIXED      confident items disagree   -> don't tilt, ask user to separate
        ASK_HUMAN  any item is low-confidence -> don't tilt, ask a human

Known limit: items that touch merge into one blob, and the classifier usually
labels a merged wet+dry blob confidently as one class (about 80% unsafe tilts
on synthetic scenes). Handling touching items needs an object detector.

Live demo (webcam; press 'r' with the plate empty to grab the reference, 'q' to quit):
  python multi_item.py --webcam --classifier_dir ../../ai_dustbin_2

Single image pair:
  python multi_item.py --reference empty.jpg --frame scene.jpg --classifier_dir ../../ai_dustbin_2
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


@dataclass
class Decision:
    action: str                    # EMPTY, TILT:DRY, TILT:WET, MIXED, ASK_HUMAN
    items: list
    reason: str


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


def item_crop(frame_bgr, item, pad=0.1):
    """Square-ish crop around the item with everything outside its mask grayed out."""
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = item.box
    p = int(max(bw, bh) * pad)
    x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(w, x + bw + p), min(h, y + bh + p)
    crop = frame_bgr[y0:y1, x0:x1].copy()
    crop[~item.mask[y0:y1, x0:x1]] = GRAY
    return crop


def classify_items(frame_bgr, items, classifier):
    """Fill in label/confidence for each item using a WasteClassifier."""
    for item in items:
        result = classifier.classify_image(item_crop(frame_bgr, item))
        item.label = "wet" if "wet" in result["label"] else "dry"
        item.confidence = result["confidence"]
    return items


def decide(items, threshold=0.6):
    if not items:
        return Decision("EMPTY", items, "nothing on the plate")
    unsure = [it for it in items if it.confidence < threshold]
    labels = {it.label for it in items if it.confidence >= threshold}
    if len(labels) > 1:
        return Decision("MIXED", items, f"{len(items)} items, both wet and dry")
    if unsure:
        return Decision("ASK_HUMAN", items, f"{len(unsure)} of {len(items)} item(s) below {threshold:.0%} confidence")
    label = labels.pop()
    return Decision(f"TILT:{label.upper()}", items, f"{len(items)} item(s), all {label}")


def draw(frame_bgr, decision):
    out = frame_bgr.copy()
    colours = {"dry": (255, 160, 0), "wet": (0, 170, 0)}
    for it in decision.items:
        x, y, bw, bh = it.box
        c = colours.get(it.label, (0, 0, 255))
        cv2.rectangle(out, (x, y), (x + bw, y + bh), c, 2)
        cv2.putText(out, f"{it.label} {it.confidence:.0%}", (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    banner = {"MIXED": (0, 0, 255), "ASK_HUMAN": (0, 140, 255)}.get(decision.action, (40, 40, 40))
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), banner, -1)
    cv2.putText(out, f"{decision.action}: {decision.reason}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def load_classifier(classifier_dir, model_path=None):
    sys.path.insert(0, os.path.abspath(classifier_dir))
    from waste_classifier import WasteClassifier
    return WasteClassifier(model_path=model_path or os.path.join(classifier_dir, "models", "waste_classifier.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-item detection on the bin plate")
    parser.add_argument("--classifier_dir", required=True, help="Folder with waste_classifier.py (the ai_dustbin_2 repo)")
    parser.add_argument("--model_path",     default=None,  help="Defaults to <classifier_dir>/models/waste_classifier.pt")
    parser.add_argument("--threshold",      type=float, default=0.6)
    parser.add_argument("--webcam",         action="store_true")
    parser.add_argument("--camera",         type=int, default=0)
    parser.add_argument("--reference",      help="Photo of the empty plate")
    parser.add_argument("--frame",          help="Photo with items on the plate")
    parser.add_argument("--save",           help="Where to save the annotated frame (image mode)")
    args = parser.parse_args()

    classifier = load_classifier(args.classifier_dir, args.model_path)
    detector = ItemDetector()

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
                decision = decide(classify_items(frame, detector.detect(frame), classifier), args.threshold)
                shown = draw(frame, decision)
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
        decision = decide(classify_items(frame, detector.detect(frame), classifier), args.threshold)
        print(f"{decision.action}: {decision.reason}")
        for it in decision.items:
            print(f"  {it.label:3s} {it.confidence:.0%}  box={it.box}  area={it.area_frac:.1%}")
        if args.save:
            cv2.imwrite(args.save, draw(frame, decision))
            print(f"Saved {args.save}")
