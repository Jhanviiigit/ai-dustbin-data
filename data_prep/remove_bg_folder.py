"""
remove_bg_folder.py
-------------------
Removes the background from every image in a folder tree (rembg / U^2-Net),
replacing it with flat gray 128: the same look as dataset D and the bin's crops.
Unlike remove_background.py it needs no manifest.csv: any folder layout works.

- Keeps the folder structure: src/train/dry waste/a.jpg -> out/train/dry waste/a.jpg
- Saves each mask to out/masks/<same path>.png (white = object)
- Drops images where the cut-out erased the object (mask < 1% of the image),
  listed in out/report.csv
- --crop: also crops each image to the object's box + 10% padding, the framing
  the detector uses on the bin
- Resumable: images already done are skipped if you run it again

Install:  pip install "rembg[cpu]" pillow numpy
Run:      python remove_bg_folder.py --src my_dataset --out my_dataset_gray
          python remove_bg_folder.py --src my_dataset --out my_dataset_gray --crop
"""

import argparse
import csv
import os
from collections import Counter

import numpy as np
from PIL import Image, ImageOps
from rembg import new_session, remove

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

p = argparse.ArgumentParser(description="Remove image backgrounds (gray 128), keeping the folder structure")
p.add_argument("--src", required=True, help="Folder with the images (any sub-folders)")
p.add_argument("--out", required=True, help="Output folder")
p.add_argument("--model", default="u2net", help="rembg model: u2net (default), u2netp (faster), isnet-general-use")
p.add_argument("--max_side", type=int, default=512, help="Shrink images so the longest side is at most this")
p.add_argument("--gray", type=int, default=128, help="Background gray level")
p.add_argument("--min_coverage", type=float, default=0.01, help="Drop images whose mask covers less than this fraction")
p.add_argument("--crop", action="store_true", help="Crop to the object's box + padding (the detector's framing)")
p.add_argument("--pad", type=float, default=0.10, help="Padding for --crop, as a fraction of the box's longer side")
args = p.parse_args()

src, out = os.path.abspath(args.src), os.path.abspath(args.out)
paths = []
for root, dirs, files in os.walk(src):
    if os.path.abspath(root).startswith(out):          # don't read our own output if it sits inside src
        continue
    paths += [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(EXTS)]
print(f"{len(paths)} images in {src}")

session = new_session(args.model)
report = []
for k, path in enumerate(paths, start=1):
    rel = os.path.relpath(path, src)
    stem = os.path.splitext(rel)[0]
    out_img = os.path.join(out, stem + ".jpg")
    out_mask = os.path.join(out, "masks", stem + ".png")

    if os.path.exists(out_mask) and os.path.exists(out_img):        # resume: already done
        mask = Image.open(out_mask).convert("L")
        report.append((rel, "kept", round(float(np.asarray(mask, np.float32).mean() / 255), 4)))
        continue

    full = ImageOps.exif_transpose(Image.open(path)).convert("RGB")  # apply phone rotation first
    img = full.copy()
    img.thumbnail((args.max_side, args.max_side))
    if os.path.exists(out_mask):                                     # mask from an earlier run: reuse it
        mask = Image.open(out_mask).convert("L")
    else:
        mask = remove(img, session=session, only_mask=True).convert("L")
        os.makedirs(os.path.dirname(out_mask), exist_ok=True)
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.BILINEAR)
    mask.save(out_mask)

    m = np.asarray(mask, np.float32) / 255
    coverage = float(m.mean())
    if coverage < args.min_coverage:
        report.append((rel, "dropped: object erased", round(coverage, 4)))
        continue

    ys, xs = np.nonzero(m > 0.5)
    if args.crop and len(xs):
        # Find the box on the small image, but cut it from the full-resolution photo, so
        # small items keep their detail; the crop is then shrunk to max_side like the rest.
        sx, sy = full.width / img.width, full.height / img.height
        x0, x1, y0, y1 = xs.min() * sx, (xs.max() + 1) * sx, ys.min() * sy, (ys.max() + 1) * sy
        pad = max(x1 - x0, y1 - y0) * args.pad
        box = (max(0, int(x0 - pad)), max(0, int(y0 - pad)), min(full.width, int(x1 + pad)), min(full.height, int(y1 + pad)))
        part = full.crop(box)
        part_mask = mask.resize(full.size, Image.BILINEAR).crop(box)
        result = Image.new("RGB", part.size, (args.gray,) * 3)
        result.paste(part, mask=part_mask)
        result.thumbnail((args.max_side, args.max_side))
    else:
        result = Image.new("RGB", img.size, (args.gray,) * 3)
        result.paste(img, mask=mask)
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    result.save(out_img, quality=95)
    report.append((rel, "kept", round(coverage, 4)))

    if k % 100 == 0 or k == len(paths):
        print(f"  {k}/{len(paths)}", flush=True)

os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "report.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["path", "status", "mask_coverage"])
    w.writerows(report)

# Drops per folder (e.g. train/dry waste): check they are not lopsided between classes
dropped = Counter(os.path.dirname(r[0]) for r in report if r[1] != "kept")
total = Counter(os.path.dirname(r[0]) for r in report)
print(f"\nkept {sum(r[1] == 'kept' for r in report)}, dropped {sum(r[1] != 'kept' for r in report)}")
for folder in sorted(total):
    print(f"  {folder or '.'}: dropped {dropped[folder]} of {total[folder]} ({dropped[folder] / total[folder]:.1%})")
print(f"Images in {out}, masks in {os.path.join(out, 'masks')}, details in report.csv")
