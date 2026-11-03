"""Session 2 (2026-11-02): which photos show which item. Every item is a test item (nothing trains on session 2).

  python -P items.py <session>              ->  <session>/items.csv
  python -P items.py <session> --dry_only   (before the wet groups are filled in)

<session> holds dry/, wet/ and background/ (the photos straight from the camera). Groups were read off the
contact sheets (sheets.py): each number is a photo's index in the folder's sorted file list. Consecutive shots
of one item go in one group; when unsure whether two shots are the same item, put them in one group (merging
is safe, splitting is not).
"""
import csv
import glob
import os
import sys
from collections import Counter

GROUPS = {
    "dry": [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16], [17, 18, 19], [20, 21, 22],
            [23, 24, 25], [26, 27, 28], [29, 30, 31], [32, 33, 34], [35, 36, 37], [38, 39, 40], [41, 42],
            [43, 44, 45], [46, 47], [48, 49, 50, 51, 52, 53], [54, 55, 56], [57, 58, 59], [60, 61, 62],
            [63, 64, 65], [66, 67, 68], [69, 70], [71, 72, 73], [74, 75, 76], [77, 78, 79]],
    "wet": [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19], [20, 21, 22],
            [23, 24, 25], [26, 27, 28], [29, 30, 31], [32, 33, 34], [35, 36, 37], [38, 39, 40], [41, 42, 43],
            [44, 45, 46], [47, 48, 49], [50, 51], [52, 53, 54], [55, 56, 57, 67, 68], [58, 59, 60], [61, 62, 63],
            [64, 65, 66]],
}
NOTES = {
    ("dry", 48): "box shown closed, then open",
    ("dry", 69): "two items in one photo (glue stick on a ruler), both dry",
    ("wet", 41): "yam",
    ("wet", 44): "grated carrot",
    ("wet", 55): "radish, whole (55-57) and a cut piece (67-68), kept as one item",
}

session = sys.argv[1]
dry_only = "--dry_only" in sys.argv
rows = []
for label in ("dry", "wet"):
    if label == "wet" and dry_only:
        continue
    fs = sorted(glob.glob(os.path.join(session, label, "*.jpg")))
    if not GROUPS[label]:
        raise SystemExit(f"no {label} groups yet: fill in GROUPS['{label}'] ({len(fs)} photos in {label}/)")
    flat = sorted(i for g in GROUPS[label] for i in g)
    if flat != list(range(len(fs))):
        missing = sorted(set(range(len(fs))) - set(flat))
        extra = sorted({i for i in flat if flat.count(i) > 1} | (set(flat) - set(range(len(fs)))))
        raise SystemExit(f"{label}: groups must use every photo 0..{len(fs) - 1} exactly once; "
                         f"missing {missing[:10]}, repeated or out of range {extra[:10]}")
    for k, g in enumerate(GROUPS[label]):
        for i in g:
            rows.append({"file": f"{label}/{os.path.basename(fs[i])}", "original": os.path.basename(fs[i]),
                         "item_id": f"s2_{label}_{k + 1:03d}", "label": label, "pile": "test",
                         "note": NOTES.get((label, g[0]), ""), "photos_of_item": len(g)})

with open(os.path.join(session, "items.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
items = {r["item_id"]: r["label"] for r in rows}
print("items", dict(Counter(items.values())), "photos", dict(Counter(r["label"] for r in rows)),
      "->", os.path.join(session, "items.csv"))
