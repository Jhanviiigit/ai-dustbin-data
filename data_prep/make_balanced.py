"""
make_balanced.py
----------------
Builds dataset C: the cleaned dataset with wet/dry balanced 50/50 inside
every source and every split, so the source of an image says nothing about
its class. The larger class in each (split, source) is randomly downsampled.

Keeps the train/val/test assignment from clean_dataset.py, so no duplicate
groups cross splits.

Usage:
  python make_balanced.py --src cleaned_dataset --out dataset_C
"""

import argparse
import csv
import os
import random
import shutil
from collections import Counter, defaultdict

CLASS_FOLDERS = {"dry": "dry waste", "wet": "wet waste"}

parser = argparse.ArgumentParser(description="Balance wet/dry within each source")
parser.add_argument("--src",  default="cleaned_dataset", help="Output folder of clean_dataset.py")
parser.add_argument("--out",  default="dataset_C",       help="Output folder (recreated each run)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


def long_path(path):
    path = os.path.abspath(path)
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + path
    return path


rows = list(csv.DictReader(open(os.path.join(args.src, "manifest.csv"), encoding="utf-8")))
kept = [r for r in rows if r["status"] == "keep"]

buckets = defaultdict(lambda: defaultdict(list))
for r in kept:
    buckets[(r["split"], r["source"])][r["label"]].append(r)

rng = random.Random(args.seed)
selected = []
for key in sorted(buckets):
    by_label = buckets[key]
    n = min(len(by_label["dry"]), len(by_label["wet"]))
    for label in ("dry", "wet"):
        items = sorted(by_label[label], key=lambda r: r["new_path"])
        rng.shuffle(items)
        selected += items[:n]

if os.path.exists(args.out):
    shutil.rmtree(long_path(args.out))
for r in selected:
    dest = os.path.join(args.out, r["new_path"])
    os.makedirs(long_path(os.path.dirname(dest)), exist_ok=True)
    shutil.copy2(long_path(os.path.join(args.src, r["new_path"])), long_path(dest))

with open(os.path.join(args.out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(selected)

table = Counter((r["split"], r["source"], r["label"]) for r in selected)
sources = sorted({r["source"] for r in selected})
lines = ["# Dataset C: balanced within each source", "",
         f"Built from `{args.src}` (seed {args.seed}). Each (split, source) is 50% dry / 50% wet.", "",
         f"- Images: **{len(selected)}** (of {len(kept)} in the cleaned dataset)", "",
         "| Split | " + " | ".join(f"{s} (dry / wet)" for s in sources) + " | Total |",
         "|---" * (len(sources) + 2) + "|"]
for split in ("train", "val", "test"):
    cells = [f"{table[(split, s, 'dry')]} / {table[(split, s, 'wet')]}" for s in sources]
    total = sum(v for (sp, _, _), v in table.items() if sp == split)
    lines.append(f"| {split} | " + " | ".join(cells) + f" | {total} |")
lines.append("")

with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
