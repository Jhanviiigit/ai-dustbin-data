"""
augment.py
----------
Balances the classes in pool/train with augmented copies of the minority class.

  - train only; never val or any test split
  - only as many copies as needed to bring the minority class up to the majority
    (recomputed on the whole train split each update)
  - at most 2 copies per original (cap: original + 2 = 3 versions)
  - each copy is made once, seeded from its own crop_id, so the same copy comes out
    every time; it shares its original's group_id and records parent_crop_id
  - changes: horizontal flip (half the time), rotation within +/-15 degrees (gray 128
    fill, like the bin's crops), brightness / contrast / saturation within +/-20%.
    No hue change: colour is a real wet/dry cue.
"""

import hashlib
from collections import Counter

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from pool import bucket

CAP = 2
GRAY = (128, 128, 128)


def make_copy(image, copy_id):
    rng = np.random.default_rng(int(hashlib.sha1(copy_id.encode("utf-8")).hexdigest()[:15], 16))
    out = image.convert("RGB")
    if rng.random() < 0.5:
        out = ImageOps.mirror(out)
    out = out.rotate(float(rng.uniform(-15, 15)), resample=Image.BILINEAR, fillcolor=GRAY)
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        out = enhancer(out).enhance(float(rng.uniform(0.8, 1.2)))
    return out


def augment_to_balance(pool, added_in, cap=CAP):
    """Add augmented copies to pool/train until the classes balance or the cap is reached.
    Returns the rows added."""
    train = pool.rows_in("train")
    count = Counter(r["label"] for r in train)
    if count["dry"] == count["wet"]:
        return []
    minority = min(("dry", "wet"), key=lambda lab: count[lab])
    need = abs(count["dry"] - count["wet"])
    copies = Counter(r["parent_crop_id"] for r in train if r["origin"] == "augmented")
    originals = sorted((r for r in train if r["label"] == minority and r["origin"] != "augmented"),
                       key=lambda r: bucket(r["crop_id"], 10 ** 9))
    added = []
    for round_ in range(1, cap + 1):                    # first copies for everyone, then second copies
        for r in originals:
            if need == 0:
                return added
            if copies[r["crop_id"]] >= round_:
                continue
            k = copies[r["crop_id"]] + 1
            copy_id = f"{r['crop_id']}_aug{k}"
            img = make_copy(Image.open(pool.abs_path(r)), copy_id)
            added.append(pool.add(img, copy_id, "train", r["label"], r["group_id"], "augmented",
                                  parent_crop_id=r["crop_id"], added_in=added_in))
            copies[r["crop_id"]] = k
            need -= 1
    return added
