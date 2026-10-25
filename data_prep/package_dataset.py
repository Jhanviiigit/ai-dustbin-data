"""
package_dataset.py
------------------
Zips a dataset folder (dataset_C, dataset_D, ...) for sharing as a GitHub
release asset. Images are shrunk so the longest side is at most --max_side px
(training resizes to 224 anyway), which makes the zip far smaller.

Keeps the train/val/test + 'dry waste'/'wet waste' layout, manifest.csv,
report.md, and masks/ if present. Adds DATA_LICENSE.txt with the source
credits.

Usage:
  python package_dataset.py --src dataset_C --out dataset_C.zip
"""

import argparse
import io
import os
import zipfile

from PIL import Image, ImageOps

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

LICENSE_TEXT = """This dataset is assembled from third-party sources. Credit them when you use it.

- RealWaste: S. Single, S. Iranmanesh, R. Raad, "RealWaste: A Novel Real-Life Data Set
  for Landfill Waste Classification Using Deep Learning", Information 14(12):633, 2023.
  Licensed CC BY-NC-SA 4.0 (non-commercial, share-alike). Files prefixed 'realwaste_'.
- Roboflow export "dry and wet waste - vdataset vibhaaa" (from siansusan/ai_dustbin_2),
  which includes TrashNet-style images (Thung & Yang, 2016).
- 'dataset2_' files (v1 datasets only; v2 leaves them out): origin unknown (from
  siansusan/ai_dustbin_2). Do not publish results on them without confirming the source.

Because RealWaste is CC BY-NC-SA 4.0, this combined dataset must be shared under the
same license and must not be used commercially.
"""

parser = argparse.ArgumentParser(description="Zip a dataset folder with resized images")
parser.add_argument("--src",      required=True, help="Dataset folder to package")
parser.add_argument("--out",      required=True, help="Output .zip path")
parser.add_argument("--max_side", type=int, default=512)
args = parser.parse_args()


def long_path(path):
    path = os.path.abspath(path)
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + path
    return path


root_name = os.path.basename(os.path.normpath(args.src))
count = 0
written = set()
with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_STORED) as zf:
    zf.writestr(f"{root_name}/DATA_LICENSE.txt", LICENSE_TEXT)
    for dirpath, _, files in os.walk(long_path(args.src)):
        for name in sorted(files):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, long_path(args.src)).replace(os.sep, "/")
            arc = f"{root_name}/{rel}"
            is_mask = rel.startswith("masks/")
            if name.lower().endswith(IMAGE_EXTS) and not is_mask:
                img = ImageOps.exif_transpose(Image.open(full)).convert("RGB")
                img.thumbnail((args.max_side, args.max_side))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=92)
                stem = os.path.splitext(arc)[0]
                if stem + ".jpg" in written:
                    raise ValueError(f"Two images would both become {stem}.jpg - rename one first")
                written.add(stem + ".jpg")
                zf.writestr(stem + ".jpg", buf.getvalue())
                count += 1
            else:
                # masks are PNGs and already small; csv/md copied as-is
                zf.write(full, arc, compress_type=zipfile.ZIP_DEFLATED)
            if count and count % 1000 == 0:
                print(f"  {count} images")

print(f"[Done] {count} images -> {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB)")
print("Note: image extensions are normalised to .jpg; manifest.csv still lists the original names.")
