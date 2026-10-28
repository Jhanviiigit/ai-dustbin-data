"""
remove_background.py
--------------------
Builds dataset D: every image from the cleaned dataset with its background
removed by U^2-Net (rembg, Qin et al., 2020) and replaced with flat gray.

  - Images are shrunk to at most --max_side px before segmentation (faster,
    and training resizes to 224 anyway).
  - The cut-out mask is saved to <out>/masks/ so other backgrounds (e.g.
    photos of the empty bin plate) can be pasted in later without re-running.
  - Images whose mask keeps less than --min_coverage of the frame are dropped:
    the model erased the object (common for piles and transparent items).
    The report counts these per class and source, since failures that hit
    one class more than the other would become a new shortcut.
  - Resumable: images already written are skipped, so the job can be stopped
    and restarted.

Keeps the train/val/test assignment from clean_dataset.py.

Usage:
  python remove_background.py --src cleaned_dataset --out dataset_D
"""

import argparse
import csv
import os
import time
from collections import Counter

import numpy as np
from PIL import Image, ImageOps
from rembg import new_session, remove

parser = argparse.ArgumentParser(description="Remove image backgrounds (dataset D)")
parser.add_argument("--src",          default="cleaned_dataset", help="Output folder of clean_dataset.py")
parser.add_argument("--out",          default="dataset_D",       help="Output folder (existing images are kept and skipped)")
parser.add_argument("--model",        default="u2net",           help="rembg model: u2net, u2netp, silueta, isnet-general-use")
parser.add_argument("--max_side",     type=int,   default=512)
parser.add_argument("--min_coverage", type=float, default=0.01,  help="Drop images whose mask covers less than this fraction")
parser.add_argument("--gray",         type=int,   default=128,   help="Background gray level")
parser.add_argument("--reuse_from",   default=None,              help="An earlier output folder of this script: masks for identical images (same SHA-256) are reused instead of recomputed")
args = parser.parse_args()


def long_path(path):
    path = os.path.abspath(path)
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + path
    return path


rows = [r for r in csv.DictReader(open(os.path.join(args.src, "manifest.csv"), encoding="utf-8"))
        if r["status"] == "keep"]
session = new_session(args.model)
print(f"[BG] {len(rows)} images, model '{args.model}'")

reuse = {}
if args.reuse_from:
    for old in csv.DictReader(open(os.path.join(args.reuse_from, "manifest.csv"), encoding="utf-8")):
        old_mask = os.path.join(args.reuse_from, "masks", os.path.splitext(old["new_path"])[0] + ".png")
        if os.path.exists(long_path(old_mask)):
            reuse[old["sha256"]] = old_mask
    print(f"[BG] {len(reuse)} masks available to reuse from '{args.reuse_from}'")

t0 = time.time()
done = 0
for k, r in enumerate(rows, start=1):
    stem = os.path.splitext(r["new_path"])[0]
    out_img = os.path.join(args.out, stem + ".jpg")
    out_mask = os.path.join(args.out, "masks", stem + ".png")

    if os.path.exists(long_path(out_mask)):
        mask = Image.open(long_path(out_mask))
    else:
        # rembg applies EXIF rotation internally, so apply it here too or the mask comes back rotated
        img = ImageOps.exif_transpose(Image.open(long_path(os.path.join(args.src, r["new_path"])))).convert("RGB")
        img.thumbnail((args.max_side, args.max_side))
        if r["sha256"] in reuse:
            mask = Image.open(long_path(reuse[r["sha256"]])).convert("L")
        else:
            mask = remove(img, session=session, only_mask=True).convert("L")
        if mask.size != img.size:
            mask = mask.resize(img.size, Image.BILINEAR)
        composite = Image.new("RGB", img.size, (args.gray,) * 3)
        composite.paste(img, mask=mask)
        os.makedirs(long_path(os.path.dirname(out_mask)), exist_ok=True)
        mask.save(long_path(out_mask))
        coverage = float(np.asarray(mask, dtype=np.float32).mean() / 255)
        if coverage >= args.min_coverage:
            os.makedirs(long_path(os.path.dirname(out_img)), exist_ok=True)
            composite.save(long_path(out_img), quality=95)
        done += 1

    coverage = float(np.asarray(mask, dtype=np.float32).mean() / 255)
    r["mask_coverage"] = round(coverage, 4)
    # Decide on the unrounded value, the same one used when the image was written
    r["bg_status"] = "keep" if coverage >= args.min_coverage else "drop (mask nearly empty)"
    r["bg_path"] = os.path.join(stem + ".jpg") if r["bg_status"] == "keep" else ""

    if k % 200 == 0 or k == len(rows):
        rate = (time.time() - t0) / max(done, 1)
        print(f"  {k}/{len(rows)}  ({rate:.2f} s/img, ~{rate * (len(rows) - k) / 60:.0f} min left)", flush=True)

# Images dropped here would otherwise be left on disk from an earlier run with a lower threshold
for r in rows:
    if r["bg_status"] != "keep":
        stale = long_path(os.path.join(args.out, os.path.splitext(r["new_path"])[0] + ".jpg"))
        if os.path.exists(stale):
            os.remove(stale)

fields = list(rows[0].keys())
with open(os.path.join(args.out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

kept = [r for r in rows if r["bg_status"] == "keep"]
dropped = Counter((r["source"], r["label"]) for r in rows if r["bg_status"] != "keep")
totals = Counter((r["source"], r["label"]) for r in rows)
table = Counter((r["split"], r["label"]) for r in kept)
lines = ["# Dataset D: background removed", "",
         f"Built from `{args.src}` with rembg model `{args.model}`, images shrunk to max {args.max_side}px, "
         f"background gray {args.gray}. Masks are in `masks/`.", "",
         f"- Images: **{len(kept)}** kept, **{len(rows) - len(kept)}** dropped (mask covers < {args.min_coverage:.0%} of the image)", "",
         "## Dropped by source and class (check these are not lopsided)", "",
         "| Source | Class | Dropped | Of | Rate |", "|---|---|---|---|---|"]
for key in sorted(totals):
    lines.append(f"| {key[0]} | {key[1]} | {dropped[key]} | {totals[key]} | {dropped[key] / totals[key]:.1%} |")
lines += ["", "## Kept images per split", "", "| Split | Dry | Wet | Total |", "|---|---|---|---|"]
for split in ("train", "val", "test"):
    lines.append(f"| {split} | {table[(split, 'dry')]} | {table[(split, 'wet')]} | {table[(split, 'dry')] + table[(split, 'wet')]} |")
lines += ["", "## Mask coverage (fraction of image kept), median by source and class", "",
          "| Source | Dry | Wet |", "|---|---|---|"]
for src in sorted({r["source"] for r in rows}):
    med = {lab: np.median([r["mask_coverage"] for r in rows if r["source"] == src and r["label"] == lab]) for lab in ("dry", "wet")}
    lines.append(f"| {src} | {med['dry']:.2f} | {med['wet']:.2f} |")
lines.append("")

with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
