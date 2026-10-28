"""
capture.py
----------
Collects labelled single-item crops from the bin's own camera, for training
the student on what the bin really sees.

Each save stores the item's crop exactly as the detector makes it on the bin
(item box + 10% padding, gray 128 outside the item), plus the full camera frame
and the item's mask, so crops can be re-cut later if the crop style changes.

Keys (in the camera window):
  r   capture the empty-plate reference   (clear the plate first)
  n   next physical item                  (new item ID)
  d   save the item on the plate as DRY
  w   save the item on the plate as WET
  u   undo the last save
  q   quit

Photograph each item a few times (move or turn it between saves), then press
'n' before the next item. Item IDs matter: split_bin_dataset.py keeps every
photo of one item in the same split, so test items are never seen in training.

A save is refused when the plate is still moving, when there is not exactly
one item, or when the one blob looks like several touching items.

Output: <out>/<session>/
  crops/<crop_id>.jpg    the detector's crop (what the student sees)
  frames/<crop_id>.jpg   the full camera frame
  masks/<crop_id>.png    the item's mask in the full frame
  captures.csv           one row per save

Usage:
  python capture.py --out ../../bin_captures --lighting daylight
  python capture.py --out ../../bin_captures --lighting lamp --camera 1 --session 2026-10-27-lab
"""

import argparse
import csv
import os
from datetime import datetime

import cv2
import numpy as np

from multi_item import DEFAULT_RULE, ItemDetector, check_count, item_crop

FIELDS = ["crop_id", "label", "item_id", "session", "lighting", "captured_at",
          "crop_file", "frame_file", "mask_file", "box", "area_frac"]


class CaptureSession:
    """Saves labelled crops for one session and keeps captures.csv in sync."""

    def __init__(self, out_root, session, lighting):
        self.dir = os.path.join(out_root, session)
        self.session = session
        self.lighting = lighting
        for sub in ("crops", "frames", "masks"):
            os.makedirs(os.path.join(self.dir, sub), exist_ok=True)
        self.csv_path = os.path.join(self.dir, "captures.csv")
        self.rows = []
        if os.path.exists(self.csv_path):            # resume a session
            with open(self.csv_path, newline="", encoding="utf-8") as f:
                self.rows = list(csv.DictReader(f))
        used = [int(r["item_id"].rsplit("-", 1)[1]) for r in self.rows]
        self.item_no = max(used, default=0) + 1 if self.rows else 1

    @property
    def item_id(self):
        return f"{self.session}-{self.item_no:03d}"

    def next_item(self):
        self.item_no += 1

    def counts(self):
        return {lab: sum(r["label"] == lab for r in self.rows) for lab in ("dry", "wet")}

    def save(self, frame_bgr, item, label):
        """Save one item's crop, frame and mask. Returns the new crop_id."""
        if any(r["item_id"] == self.item_id and r["label"] != label for r in self.rows):
            raise ValueError(f"{self.item_id} was already saved as the other class; press 'n' for a new item")
        n = sum(r["item_id"] == self.item_id for r in self.rows) + 1
        crop_id = f"{self.item_id}-{n:02d}"
        files = {"crop_file": f"crops/{crop_id}.jpg", "frame_file": f"frames/{crop_id}.jpg", "mask_file": f"masks/{crop_id}.png"}
        cv2.imwrite(os.path.join(self.dir, files["crop_file"]), item_crop(frame_bgr, item), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(self.dir, files["frame_file"]), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(self.dir, files["mask_file"]), item.mask.astype(np.uint8) * 255)
        row = {"crop_id": crop_id, "label": label, "item_id": self.item_id, "session": self.session,
               "lighting": self.lighting, "captured_at": datetime.now().isoformat(timespec="seconds"),
               "box": " ".join(map(str, item.box)), "area_frac": f"{item.area_frac:.4f}", **files}
        self.rows.append(row)
        self._write()
        return crop_id

    def undo(self):
        """Remove the last save. Returns its crop_id, or None."""
        if not self.rows:
            return None
        row = self.rows.pop()
        for key in ("crop_file", "frame_file", "mask_file"):
            path = os.path.join(self.dir, row[key])
            if os.path.exists(path):
                os.remove(path)
        self._write()
        return row["crop_id"]

    def _write(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)


def check_ready(detector, frame_bgr, prev_gray, motion_level=20, motion_frac=0.004):
    """(item, message): the single item to save, or None and why not."""
    gray = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.int16)
    if prev_gray is not None and (np.abs(gray - prev_gray) > motion_level).mean() > motion_frac:
        return None, "plate still moving"
    items = detector.detect(frame_bgr)
    count = check_count(frame_bgr, items, DEFAULT_RULE)
    if count.action == "EMPTY":
        return None, "no item on the plate"
    if count.action == "MULTIPLE":
        return None, f"one item at a time ({count.reason})"
    return items[0], "ready"


def overlay(frame_bgr, session, item, message):
    out = frame_bgr.copy()
    if item is not None:
        x, y, w, h = item.box
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
    c = session.counts()
    lines = [f"item {session.item_id}   saved: dry {c['dry']}  wet {c['wet']}",
             f"{message}   [d] dry  [w] wet  [n] next item  [u] undo  [r] reference  [q] quit"]
    cv2.rectangle(out, (0, 0), (out.shape[1], 52), (40, 40, 40), -1)
    for i, text in enumerate(lines):
        cv2.putText(out, text, (8, 20 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect labelled item crops from the bin camera")
    parser.add_argument("--out",      required=True, help="Folder for all capture sessions")
    parser.add_argument("--session",  default=datetime.now().strftime("%Y-%m-%d-%H%M"), help="Session name (default: date and time)")
    parser.add_argument("--lighting", required=True, help="Lighting tag for this session, e.g. daylight, lamp, dim")
    parser.add_argument("--camera",   type=int, default=0)
    args = parser.parse_args()

    session = CaptureSession(args.out, args.session, args.lighting)
    detector = ItemDetector()
    cap = cv2.VideoCapture(args.camera)
    prev_gray = None
    event, event_until = "", 0.0           # last key result, shown for 2 s
    print(f"Saving to {session.dir}")

    def show(text):
        global event, event_until
        event, event_until = text, cv2.getTickCount() / cv2.getTickFrequency() + 2.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed")
            break
        item, status = None, "clear the plate and press [r]"
        if detector.reference is not None:
            item, status = check_ready(detector, frame, prev_gray)
        prev_gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.int16)
        now = cv2.getTickCount() / cv2.getTickFrequency()
        cv2.imshow("capture", overlay(frame, session, item, event if now < event_until else status))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            detector.set_reference([cap.read()[1] for _ in range(10)])
            show("reference captured")
        elif key == ord("n"):
            session.next_item()
            show(f"next item: {session.item_id}")
        elif key == ord("u"):
            undone = session.undo()
            show(f"undid {undone}" if undone else "nothing to undo")
        elif key in (ord("d"), ord("w")):
            if item is None:
                show(f"not saved: {status}")
            else:
                try:
                    show(f"saved {session.save(frame, item, 'dry' if key == ord('d') else 'wet')}")
                except ValueError as e:
                    show(f"not saved: {e}")

    cap.release()
    cv2.destroyAllWindows()
    c = session.counts()
    print(f"Session {session.session}: dry {c['dry']}, wet {c['wet']}, saved to {session.dir}")
