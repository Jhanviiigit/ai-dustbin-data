"""Contact sheets of a photo folder, numbered in sorted file order, to group the photos into items by eye.

  python -P sheets.py <session>/wet <out folder> wet      ->  <out>/wet_00.jpg, wet_01.jpg, ...

The number on each tile is the index items.py uses.
"""
import glob
import os
import sys

from PIL import Image, ImageDraw

src, out, tag = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
fs = sorted(glob.glob(os.path.join(src, "*.jpg")))
W, H, C, per = 200, 150, 6, 30
for p in range(0, len(fs), per):
    chunk = fs[p:p + per]
    rows = (len(chunk) + C - 1) // C
    sheet = Image.new("RGB", (C * W, rows * (H + 16)), "white")
    d = ImageDraw.Draw(sheet)
    for i, f in enumerate(chunk):
        x, y = (i % C) * W, (i // C) * (H + 16)
        sheet.paste(Image.open(f).convert("RGB").resize((W, H)), (x, y + 16))
        name = os.path.basename(f).replace("nvcamtest_", "").replace("_s00_", "/").replace(".jpg", "")
        d.text((x + 3, y + 2), f"{p + i}: {name}", fill="black")
    sheet.save(os.path.join(out, f"{tag}_{p // per:02d}.jpg"), quality=85)
print(f"{tag}: {len(fs)} photos -> {out}")
