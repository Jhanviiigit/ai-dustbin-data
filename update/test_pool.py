"""
test_pool.py
------------
Unit tests for update/pool.py: the pool's rules hold, and a saved pool reloads
exactly. No datasets needed (images are generated).

  python update/test_pool.py
"""

import csv
import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool import COLUMNS, Pool, PoolError, bucket, hamming, phash, seed_from_organik

failures = []


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + what)
    if not cond:
        failures.append(what)


def refused(fn, what):
    try:
        fn()
        check(False, f"refused: {what}")
    except PoolError:
        check(True, f"refused: {what}")


rng = np.random.default_rng(0)


def image(seed):
    r = np.random.default_rng(seed)
    a = np.full((96, 96, 3), 128, np.uint8)
    a[20:76, 20:76] = r.integers(0, 255, 3)
    a[30:60, 30:50] = r.integers(0, 255, 3)
    return Image.fromarray(a)


root = tempfile.mkdtemp(prefix="pool_test_")
src = os.path.join(root, "a.jpg")
image(1).save(src, quality=95)
pool = Pool(os.path.join(root, "pool"))

print("Adding rows:")
r = pool.add(src, "c1", "train", "dry", "item-1", "capture", added_in="seed")
check(r["path"] == "train/dry waste/c1.jpg" and os.path.exists(pool.abs_path(r)), "real crop copied to train/dry waste/c1.jpg")
check(open(src, "rb").read() == open(pool.abs_path(r), "rb").read(), "a .jpg source keeps its exact bytes")
pool.add(image(2), "c2", "train", "dry", "item-1", "capture")
pool.add(image(3), "f1", "test_audit", "wet", "load-9", "feedback", trigger="audit", from_drop=1, student_conf=0.93)
pool.add(image(4), "c1_aug1", "train", "dry", "item-1", "augmented", parent_crop_id="c1")
check(len(pool.rows) == 4 and pool.split_of_group("item-1") == "train", "rows added, group remembers its split")

print("Rules:")
refused(lambda: pool.add(image(5), "c1", "train", "dry", "item-2", "capture"), "duplicate crop_id")
refused(lambda: pool.add(image(5), "x1", "train", "dry waste", "item-2", "capture"), "label 'dry waste' (must be dry/wet)")
refused(lambda: pool.add(image(5), "x2", "train", "dry", "item-2", "feedback", trigger="low_conf"), "trigger 'low_conf'")
refused(lambda: pool.add(image(5), "x3", "test", "dry", "item-2", "capture"), "unknown split 'test'")
refused(lambda: pool.add(image(5), "x4", "val", "dry", "item-1", "capture"), "moving group item-1 from train to val")
refused(lambda: pool.add(image(5), "x5", "test_audit", "wet", "load-9", "augmented", parent_crop_id="f1"), "augmented copy in a test split")
refused(lambda: pool.add(image(5), "x6", "train", "dry", "item-1", "augmented", parent_crop_id="nope"), "augmented copy without a real parent")
refused(lambda: pool.add(image(5), "x7", "train", "wet", "item-1", "augmented", parent_crop_id="c1"), "augmented copy with another label")
refused(lambda: pool.add(image(5), "x8", "train", "dry", "item-3", "capture", parent_crop_id="c1"), "parent_crop_id on a non-augmented row")
check(len(pool.rows) == 4, "a refused add changes nothing")

print("Save and reload:")
pool.save()
with open(pool.manifest_path, newline="", encoding="utf-8") as f:
    header = next(csv.reader(f))
check(header == COLUMNS, "manifest.csv has exactly the agreed columns, in order")
again = Pool(pool.root)
check([dict(r) for r in again.rows] == [dict(r) for r in pool.rows], "reloaded manifest equals what was written")
check(again.split_of_group("load-9") == "test_audit", "reloaded pool remembers group splits")
check(again.check() == [], "check() finds no problems")
os.remove(again.abs_path(again.by_id["c2"]))
check(any("file missing" in p for p in again.check()), "check() reports a missing file")

print("Organik seed (dataset_organik_v2 layout):")


def organik(folder, rows):
    """rows: [(split, class_folder, filename, group)] -> a dataset_organik_v2-style folder."""
    os.makedirs(folder)
    with open(os.path.join(folder, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "class_folder", "filename", "new_path", "source", "group", "group_v1", "source_folder"])
        for i, (split, cls, name, group) in enumerate(rows):
            new_path = f"{split}/{cls}/{name}"
            os.makedirs(os.path.join(folder, split, cls), exist_ok=True)
            image(100 + i).save(os.path.join(folder, *new_path.split("/")), quality=95)
            w.writerow([split, cls, name, new_path, "organik", group, group, "Organic"])
    return folder


og = organik(os.path.join(root, "og"), [("train", "wet waste", "a1.jpg", 0), ("train", "wet waste", "a2.jpg", 0),
                                         ("train", "dry waste", "b1.jpg", 1), ("val", "dry waste", "c1.jpg", 2),
                                         ("test", "wet waste", "d1.jpg", 3)])
op = Pool(os.path.join(root, "opool"))
added = seed_from_organik(op, og)
check(dict(added) == {"train": 3, "val": 1, "test_fixed": 1}, f"train -> train, val -> val, test -> test_fixed {dict(added)}")
r = op.by_id["organik_a1"]
check(r["origin"] == "public" and r["group_id"] == "organik_g0" and r["label"] == "wet" and r["added_in"] == "seed",
      "origin public, group_id organik_g<group>, label from the class folder")
check(op.check() == [] and op.split_of_group("organik_g3") == "test_fixed", "seeded pool passes check()")
check(seed_from_organik(op, og) == {}, "seeding again adds nothing")
bad = organik(os.path.join(root, "og_bad"), [("train", "wet waste", "e1.jpg", 7), ("test", "wet waste", "e2.jpg", 7)])
refused(lambda: seed_from_organik(Pool(os.path.join(root, "bpool")), bad), "an Organik group in both train and test")
dup = organik(os.path.join(root, "og_dup"), [("train", "wet waste", "f1.jpg", 8), ("train", "dry waste", "f1.jpg", 9)])
refused(lambda: seed_from_organik(Pool(os.path.join(root, "dpool")), dup), "the same file name in two class folders")

print("Helpers:")
check(phash(image(1)) == phash(image(1)) and len(phash(image(1))) == 16, "phash is stable, 16 hex digits")
check(hamming(phash(image(1)), phash(image(1))) == 0, "hamming of identical images is 0")
check(bucket("load-9", 100) == bucket("load-9", 100) and 0 <= bucket("abc", 5) < 5, "bucket is stable and in range")

shutil.rmtree(root, ignore_errors=True)
print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}")
sys.exit(1 if failures else 0)
