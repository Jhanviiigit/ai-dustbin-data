"""
test_capture.py
---------------
End-to-end check of capture.py + split_bin_dataset.py without a camera:
simulates capture sessions on synthetic plate frames (synth.py), then builds
dataset_bin and checks it.

Checks:
  - a frame with two items is refused; an empty plate is refused
  - undo removes the files and the CSV row
  - saving one item under both labels is refused
  - every photo of an item lands in one split, and both classes appear in every split
  - files in manifest.csv exist; crops are gray (128) around the item
  - --test_items pins those items to test

Usage:
  python test_capture.py --clean ../../cleaned_v2 --masks ../../dataset_D_v2/masks
"""

import argparse
import csv
import os
import random
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from capture import CaptureSession, check_ready
from multi_item import ItemDetector
from synth import ItemPool, SceneMaker

parser = argparse.ArgumentParser()
parser.add_argument("--clean", required=True, help="cleaned dataset folder (images + manifest.csv)")
parser.add_argument("--masks", required=True, help="dataset D masks folder for the same dataset")
parser.add_argument("--items", type=int, default=12, help="Items per class per session")
args = parser.parse_args()

pool = ItemPool(args.clean, args.masks, "train", seed=1)
maker = SceneMaker(pool, seed=1)
rng = random.Random(1)
work = tempfile.mkdtemp(prefix="capture_test_")
captures = os.path.join(work, "captures")
failures = []


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + what)
    if not cond:
        failures.append(what)


print(f"Working in {work}")
for s_no, lighting in enumerate(("daylight", "lamp")):
    session = CaptureSession(captures, f"sess{s_no}", lighting)
    for label in ("dry", "wet"):
        for _ in range(args.items):
            cutout = maker.pool.pick(label)
            for _ in range(rng.randint(2, 4)):               # a few photos of the same item
                plate = maker.plate()
                scene = plate.copy()
                maker.paste(scene, cutout, rng.randint(200, 440), rng.randint(160, 320), rng.randint(140, 200))
                detector = ItemDetector()
                detector.set_reference(maker.snap(plate))
                frame = maker.snap(scene)
                item, status = check_ready(detector, frame, None)
                if item is not None:
                    session.save(frame, item, label)
            session.next_item()

    if s_no == 0:
        print("Refusals and undo:")
        plate = maker.plate()
        detector = ItemDetector()
        detector.set_reference(maker.snap(plate))
        check(check_ready(detector, maker.snap(plate), None)[0] is None, "empty plate is refused")
        two = plate.copy()
        maker.paste(two, maker.pool.pick("dry"), 160, 140, 170)
        maker.paste(two, maker.pool.pick("wet"), 480, 340, 170)
        item, status = check_ready(detector, maker.snap(two), None)
        check(item is None and "one item" in status, f"two items are refused ({status})")

        one = plate.copy()
        maker.paste(one, maker.pool.pick("dry"), 320, 240, 170)
        frame = maker.snap(one)
        item, _ = check_ready(detector, frame, None)
        n_before = len(session.rows)
        crop_id = session.save(frame, item, "dry")
        crop_path = os.path.join(session.dir, "crops", f"{crop_id}.jpg")
        crop = cv2.imread(crop_path)
        corners = np.concatenate([crop[:3, :3].reshape(-1, 3), crop[-3:, -3:].reshape(-1, 3)])
        check(np.abs(corners.astype(int) - 128).max() <= 6, "crop corners are gray 128 (outside the item)")
        try:
            session.save(frame, item, "wet")
            check(False, "same item under both labels is refused")
        except ValueError:
            check(True, "same item under both labels is refused")
        session.undo()
        check(len(session.rows) == n_before and not os.path.exists(crop_path), "undo removes the row and the files")
        session.next_item()

print("Split by item:")
first_items = sorted({r["item_id"] for r in CaptureSession(captures, "sess0", "daylight").rows})[:3]
pinned = os.path.join(work, "test_items.txt")
with open(pinned, "w") as f:
    f.write("\n".join(first_items))
out = os.path.join(work, "dataset_bin")
here = os.path.dirname(os.path.abspath(__file__))
result = subprocess.run([sys.executable, os.path.join(here, "split_bin_dataset.py"), "--captures", captures,
                         "--out", out, "--test_items", pinned], capture_output=True, text=True)
print(result.stdout)
check(result.returncode == 0, "split_bin_dataset.py ran" + ("" if result.returncode == 0 else f": {result.stderr[-300:]}"))

manifest = list(csv.DictReader(open(os.path.join(out, "manifest.csv"), encoding="utf-8")))
splits_per_item = {}
for m in manifest:
    splits_per_item.setdefault(m["item_id"], set()).add(m["split"])
check(all(len(s) == 1 for s in splits_per_item.values()), f"each of {len(splits_per_item)} items is in exactly one split")
check(all(os.path.exists(os.path.join(out, m["new_path"])) for m in manifest), f"all {len(manifest)} manifest files exist")
check(all(splits_per_item[i] == {"test"} for i in first_items), "--test_items are in test")
present = {(m["split"], m["label"]) for m in manifest}
check(all((s, l) in present for s in ("train", "val", "test") for l in ("dry", "wet")), "both classes in every split")
layout = all(os.path.isdir(os.path.join(out, s, c)) for s in ("train", "val", "test") for c in ("dry waste", "wet waste"))
check(layout, "folder layout matches the student's bin variant")

print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}  (files left in {work})")
sys.exit(1 if failures else 0)
