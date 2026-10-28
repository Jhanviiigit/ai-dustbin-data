"""Split the Organik/Anorganik gray crops into train/val/test, grouping photos of the same item.

Step 1 of 2. The groups here (perceptual hash + file order) miss the same item in another
pose, so this split leaks and its scores are too good. Use the output only as the input of
regroup_organik.py, which regroups by image similarity and writes the split to train on.

Input: a folder with `Organic/` and `Inorganic/` (organik_gray_crops.zip, background gray 128).
Organic -> `wet waste`, Inorganic -> `dry waste`.

Photos of one item are shot in a row, so near-duplicates must stay in one split. Two photos
are put in one group when
  - their perceptual hashes are close (Hamming <= --phash_dist of 64), or
  - they are neighbours in file order (same class) and the colours of their item pixels
    are alike (histogram intersection >= --colour_sim).
Groups are never split across train/val/test. A group that holds both classes (the same item
filed under Organic and Inorganic) is a label conflict and is left out entirely.

Balance: val and test hold the same number of images per class (--eval_frac of the
smaller class each); everything else goes to train, which keeps the larger class whole
(the student's loss is class-weighted). Images over the val/test quota are discarded,
not moved to train, so nothing leaks.

  python data_prep/split_organik.py --src grey/organik --out dataset_organik
"""
import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--src", required=True, help="folder with Organic/ and Inorganic/")
parser.add_argument("--out", required=True, help="output folder (train/val/test/<class>/ + manifest.csv + report.md)")
parser.add_argument("--eval_frac", type=float, default=0.15, help="val and test size, as a share of the smaller class, each")
parser.add_argument("--phash_dist", type=int, default=4, help="max Hamming distance (of 64 bits) for near-duplicates")
parser.add_argument("--colour_sim", type=float, default=0.6, help="min colour-histogram intersection for neighbouring files")
parser.add_argument("--max_group", type=int, default=12, help="a chain of neighbours is cut into groups of at most this size")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

FOLDERS = {"Organic": "wet waste", "Inorganic": "dry waste"}
GRAY = 128


def _dct_matrix(n):
    k, i = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    m[0] *= 1 / np.sqrt(2)
    return m * np.sqrt(2 / n)


DCT32 = _dct_matrix(32)


def phash(gray_img):
    """64-bit perceptual hash (same as clean_dataset.py)."""
    pixels = np.asarray(gray_img.resize((32, 32), Image.LANCZOS), dtype=np.float64)
    low = (DCT32 @ pixels @ DCT32.T)[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


def colour_hist(rgb):
    """Normalised 4x4x4 RGB histogram of the item pixels (everything that is not the gray fill)."""
    a = np.asarray(rgb, dtype=np.int16)
    item = a[np.abs(a - GRAY).max(axis=2) > 12]
    if len(item) < 50:
        item = a.reshape(-1, 3)
    h = np.bincount(((item // 64) * np.array([16, 4, 1])).sum(1), minlength=64).astype(np.float64)
    return h / h.sum()


files = []        # (class folder, file name, path)
for folder in FOLDERS:
    for p in sorted((Path(args.src) / folder).glob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            files.append((folder, p.name, p))
if not files:
    raise SystemExit(f"no images under {args.src}/Organic and /Inorganic")

hashes, hists = [], []
for _, _, p in files:
    with Image.open(p) as im:
        im = im.convert("RGB")
        hashes.append(phash(im.convert("L")))
        hists.append(colour_hist(im))
hashes = np.array(hashes, dtype=np.uint64)

parent = list(range(len(files)))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def union(i, j):
    parent[find(i)] = find(j)


# 1. near-duplicates anywhere (either class: a hash match across classes would be a labelling problem)
popcount = np.vectorize(lambda v: bin(int(v)).count("1"))
for i in range(len(files)):
    d = popcount(hashes[i + 1:] ^ hashes[i]) if i + 1 < len(files) else []
    for k in np.nonzero(np.array(d) <= args.phash_dist)[0]:
        union(i, i + 1 + int(k))

# 2. neighbours in file order with alike item colours, as chains cut to --max_group
chain_start = 0
for i in range(1, len(files)):
    same = files[i][0] == files[i - 1][0]
    sim = np.minimum(hists[i], hists[i - 1]).sum()
    if not same or sim < args.colour_sim or i - chain_start >= args.max_group:
        chain_start = i
    else:
        union(i, i - 1)

groups = defaultdict(list)
for i in range(len(files)):
    groups[find(i)].append(i)
group_ids = {root: n for n, root in enumerate(sorted(groups, key=lambda r: min(groups[r])))}
mixed = [g for g in groups.values() if len({files[i][0] for i in g}) > 1]
if mixed:
    print(f"warning: {len(mixed)} group(s) hold the same-looking item under both classes (label conflict); they are left out")
    for g in mixed:
        print("  mixed group:", [f"{files[i][0]}/{files[i][1]}" for i in g][:6])

rng = random.Random(args.seed)
split_of = {}
class_groups = defaultdict(list)
for root, members in groups.items():
    if len({files[i][0] for i in members}) > 1:
        continue                                        # conflicting labels: out of every split
    majority = Counter(files[i][0] for i in members).most_common(1)[0][0]
    class_groups[majority].append(root)

n_class = Counter(f[0] for f in files)
quota = max(1, round(args.eval_frac * min(n_class.values())))      # per class, for val and for test
kept = set()
for folder, roots in class_groups.items():
    rng.shuffle(roots)
    filled = {"test": 0, "val": 0}
    for root in roots:
        members = groups[root]
        want = next((s for s in ("test", "val") if filled[s] < quota), "train")
        if want != "train":
            take = members[: quota - filled[want]]          # trim the last group to the quota; the rest is discarded
            filled[want] += len(take)
            kept.update((i, want) for i in take)
            for i in members:
                split_of[i] = want
        else:
            kept.update((i, "train") for i in members)
            for i in members:
                split_of[i] = "train"

out = Path(args.out)
if out.exists():
    raise SystemExit(f"{out} exists; remove it or pick another --out")
rows = []
for i, split in sorted(kept):
    folder, name, p = files[i]
    if len(Path(name).stem) > 40:                      # phone-gallery names are long; keep paths short on Windows
        name = f"{Path(name).stem[-30:]}{Path(name).suffix}"
    dest = out / split / FOLDERS[folder]
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dest / name)
    rows.append({"split": split, "class_folder": FOLDERS[folder], "filename": name,
                 "new_path": f"{split}/{FOLDERS[folder]}/{name}", "source": "organik",   # new_path, source: read by student/evaluate.py
                 "group": group_ids[find(i)], "source_folder": folder})
with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["split", "class_folder", "filename", "new_path", "source", "group", "source_folder"])
    w.writeheader()
    w.writerows(rows)

counts = Counter((r["split"], r["class_folder"]) for r in rows)
sizes = Counter(len(g) for g in groups.values())
lines = ["# Organik split", "",
         f"- Source: {args.src} ({len(files)} images: {dict(n_class)})",
         f"- Groups: {len(groups)} (sizes: " + ", ".join(f"{s}x{n}" for s, n in sorted(sizes.items())) + ")",
         f"- Seed {args.seed}; near-duplicate pHash distance <= {args.phash_dist}; neighbour colour similarity >= {args.colour_sim}; chains cut at {args.max_group}",
         f"- val and test: {quota} images per class each; images over the quota or in conflicting groups were left out: "
         f"{len(files) - len(rows)}", "",
         "| Split | dry waste | wet waste | Total |", "|---|---|---|---|"]
for s in ("train", "val", "test"):
    d, wv = counts[(s, "dry waste")], counts[(s, "wet waste")]
    lines.append(f"| {s} | {d} | {wv} | {d + wv} |")
(out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
