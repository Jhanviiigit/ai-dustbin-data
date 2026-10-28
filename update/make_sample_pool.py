"""
make_sample_pool.py
-------------------
Builds a SAMPLE pool from a dataset_bin and a feedback folder, so the teacher and
student readers can be written against a real example before the full pipeline
exists. It uses the agreed routing in its simplest form:

  dataset_bin  train -> train, val -> val, test -> test_fixed      (group = item_id)
  feedback     audit                          -> test_audit        (group = load_id)
               low_confidence, bucket(group, 5) == 0  -> test_hard
               low_confidence, bucket(group, 100) < 15 -> val
               other low_confidence           -> train
  augmented    the minority class among new train feedback gets up to 2 flipped /
               brightened copies per item (parent_crop_id set), up to balance

The real pipeline (split.py, augment.py, leak checks, trigger) replaces this
routing; the pool format is the same.

  python update/make_sample_pool.py --dataset_bin dataset_bin --feedback feedback --out sample_pool
"""

import argparse
import csv
import os
import shutil
from collections import Counter

from PIL import Image, ImageEnhance, ImageOps

from pool import Pool, bucket, seed_from_dataset_bin

ap = argparse.ArgumentParser(description="Build a sample pool from dataset_bin + a feedback folder")
ap.add_argument("--dataset_bin", required=True)
ap.add_argument("--feedback", required=True, help="hitl.py FeedbackStore folder")
ap.add_argument("--out", required=True)
args = ap.parse_args()

if os.path.exists(args.out):
    shutil.rmtree(args.out)
pool = Pool(args.out)
print("seed:", dict(seed_from_dataset_bin(pool, args.dataset_bin)))

with open(os.path.join(args.feedback, "feedback.csv"), newline="", encoding="utf-8") as f:
    feedback = [r for r in csv.DictReader(f) if r["human_label"] in ("dry", "wet")]
new_train = []
for r in feedback:
    group = r["load_id"]
    split = pool.split_of_group(group)
    if split is None:
        if r["trigger"] == "audit":
            split = "test_audit"
        elif bucket(group, 5) == 0:
            split = "test_hard"
        elif bucket(group, 100) < 15:
            split = "val"
        else:
            split = "train"
    row = pool.add(os.path.join(args.feedback, *r["crop_file"].split("/")), r["crop_id"], split, r["human_label"],
                   group, "feedback", trigger=r["trigger"], from_drop=r.get("from_drop", ""),
                   added_in="sample_update_1", student_conf=r["student_confidence"])
    if split == "train":
        new_train.append(row)

# Augment the minority class of the new train feedback, up to balance, max 2 copies per item
count = Counter(r["label"] for r in new_train)
if len(count) == 2 and count["dry"] != count["wet"]:
    minority = min(count, key=count.get)
    need = abs(count["dry"] - count["wet"])
    for r in [x for x in new_train if x["label"] == minority]:
        for k in (1, 2):
            if need == 0:
                break
            img = Image.open(pool.abs_path(r)).convert("RGB")
            img = ImageOps.mirror(img) if k == 1 else ImageEnhance.Brightness(img).enhance(1.15)
            pool.add(img, f"{r['crop_id']}_aug{k}", "train", r["label"], r["group_id"], "augmented",
                     parent_crop_id=r["crop_id"], added_in="sample_update_1")
            need -= 1

pool.save()
problems = pool.check()
print("pool:", dict(sorted(Counter(r["split"] for r in pool.rows).items())),
      "| origins:", dict(Counter(r["origin"] for r in pool.rows)))
print("\n".join(problems) if problems else "pool OK: no rule violations")
raise SystemExit(1 if problems else 0)
