"""
split_bin_dataset.py
--------------------
Turns capture sessions (capture.py) into dataset_bin, the student's training
data from the bin's own camera.

Splits by ITEM, not by photo: every photo of one physical item goes into the
same split, so the test set only holds items the model never saw. Items are
assigned per class towards the target ratios (by photo count). Items listed in
--test_items always go to test, so a fixed set of test items can be kept
across re-splits as more data arrives.

Output (the layout student/ expects for its "bin" variant):
  dataset_bin/
    train/  dry waste/  wet waste/      <crop_id>.jpg
    val/    dry waste/  wet waste/
    test/   dry waste/  wet waste/
    masks/<split>/<class>/<crop_id>.png  (mask in the full frame, for re-cropping later)
    manifest.csv   crop_id, new_path, split, label, source, item_id, session, lighting, captured_at, frame_file
    report.md      photos and items per split and class

Usage:
  python split_bin_dataset.py --captures ../../bin_captures --out ../../dataset_bin
  python split_bin_dataset.py --captures ../../bin_captures --out ../../dataset_bin --test_items test_items.txt --overwrite
"""

import argparse
import csv
import glob
import os
import random
import shutil
from collections import Counter, defaultdict

CLASS_FOLDERS = {"dry": "dry waste", "wet": "wet waste"}
SPLITS = ("train", "val", "test")

parser = argparse.ArgumentParser(description="Split capture sessions into dataset_bin, by item")
parser.add_argument("--captures",   required=True, nargs="+", help="Capture folder(s); every */captures.csv inside is used")
parser.add_argument("--out",        required=True, help="Output dataset folder")
parser.add_argument("--ratios",     type=float, nargs=3, default=[0.70, 0.15, 0.15], help="train val test")
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--test_items", default=None, help="Text file with one item_id per line that must go to test")
parser.add_argument("--overwrite",  action="store_true", help="Replace --out if it exists")
args = parser.parse_args()

# ─── Gather every capture ─────────────────────────────────────────────────────
rows = []
for root in args.captures:
    for csv_path in sorted(glob.glob(os.path.join(root, "*", "captures.csv"))):
        session_dir = os.path.dirname(csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["_dir"] = session_dir
                rows.append(r)
if not rows:
    raise SystemExit(f"No captures.csv found under {args.captures}")

items = defaultdict(list)
for r in rows:
    items[r["item_id"]].append(r)
bad = [i for i, rs in items.items() if len({r["label"] for r in rs}) > 1]
if bad:
    raise SystemExit(f"Items saved with both labels (fix their captures.csv): {bad}")

forced_test = set()
if args.test_items:
    with open(args.test_items, encoding="utf-8") as f:
        forced_test = {line.strip() for line in f if line.strip()}
    missing = forced_test - set(items)
    if missing:
        print(f"warning: {len(missing)} --test_items not found in the captures: {sorted(missing)[:5]}")

# ─── Assign whole items to splits, per class ─────────────────────────────────
rng = random.Random(args.seed)
ratios = [x / sum(args.ratios) for x in args.ratios]
split_of = {}
for label in CLASS_FOLDERS:
    ids = sorted(i for i, rs in items.items() if rs[0]["label"] == label)
    total = sum(len(items[i]) for i in ids)
    filled = dict.fromkeys(SPLITS, 0)
    for i in ids:
        if i in forced_test:
            split_of[i] = "test"
            filled["test"] += len(items[i])
    rest = [i for i in ids if i not in forced_test]
    rng.shuffle(rest)
    for i in rest:
        n = len(items[i])
        split = min(SPLITS, key=lambda s: (filled[s] + n) / max(ratios[SPLITS.index(s)] * total, 1e-9))
        split_of[i] = split
        filled[split] += n

# ─── Write the dataset ───────────────────────────────────────────────────────
if os.path.exists(args.out):
    if not args.overwrite:
        raise SystemExit(f"{args.out} exists; pass --overwrite to rebuild it")
    shutil.rmtree(args.out)

manifest = []
for item_id, rs in sorted(items.items()):
    split = split_of[item_id]
    for r in rs:
        cls = CLASS_FOLDERS[r["label"]]
        new_path = f"{split}/{cls}/{r['crop_id']}.jpg"
        os.makedirs(os.path.join(args.out, split, cls), exist_ok=True)
        shutil.copy2(os.path.join(r["_dir"], r["crop_file"]), os.path.join(args.out, new_path))
        mask_src = os.path.join(r["_dir"], r["mask_file"])
        if os.path.exists(mask_src):
            os.makedirs(os.path.join(args.out, "masks", split, cls), exist_ok=True)
            shutil.copy2(mask_src, os.path.join(args.out, "masks", split, cls, f"{r['crop_id']}.png"))
        manifest.append({"crop_id": r["crop_id"], "new_path": new_path, "split": split, "label": r["label"],
                         "source": "bin", "item_id": item_id, "session": r["session"], "lighting": r["lighting"],
                         "captured_at": r["captured_at"],
                         "frame_file": os.path.relpath(os.path.join(r["_dir"], r["frame_file"]), args.out).replace("\\", "/")})

with open(os.path.join(args.out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
    writer.writeheader()
    writer.writerows(manifest)

photos = Counter((m["split"], m["label"]) for m in manifest)
n_items = Counter((split_of[i], rs[0]["label"]) for i, rs in items.items())
lines = ["# dataset_bin", "",
         f"From {len({r['session'] for r in rows})} capture session(s), {len(items)} items, {len(manifest)} photos. "
         f"Split by item (seed {args.seed}, ratios {args.ratios}); {len(forced_test & set(items))} items pinned to test.", "",
         "| Split | Dry photos (items) | Wet photos (items) | Total photos |", "|---|---|---|---|"]
for s in SPLITS:
    lines.append(f"| {s} | {photos[(s, 'dry')]} ({n_items[(s, 'dry')]}) | {photos[(s, 'wet')]} ({n_items[(s, 'wet')]}) | "
                 f"{photos[(s, 'dry')] + photos[(s, 'wet')]} |")
lighting = Counter((m["split"], m["lighting"]) for m in manifest)
lines += ["", "Photos per lighting: " + ", ".join(f"{s}/{l}: {n}" for (s, l), n in sorted(lighting.items())), ""]
with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
