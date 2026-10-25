"""
clean_dataset.py
----------------
Cleans the wet/dry waste dataset in ai_dustbin_2 and writes a fresh
train/val/test split. The original train/ and val/ folders are not touched.

What it does:
  1. Scans train/ and val/ (or valid/) for images, labels them dry/wet and
     tags each with its source ('roboflow' or 'dataset2'). With --realwaste,
     also adds the RealWaste dataset (Single et al., 2023, CC BY-NC-SA 4.0),
     mapping its 9 classes to dry/wet (see REALWASTE_MAP).
  2. Drops unreadable, tiny and near-blank images.
  3. Finds exact duplicates (SHA-256) and near-duplicates (perceptual hash).
     Groups whose copies disagree on the label are dropped. Exact copies are
     reduced to one. Near-duplicates (burst shots of the same object, re-crops)
     are kept, but each group goes into a single split, so no near-copy of a
     test image is ever trained on. Pass --drop_near to keep one per group.
  4. Splits 70/15/15, stratified by class AND source, so every split has the
     same mix of sources and the val/test sets share no images with train.
  5. Writes <out>/{train,val,test}/{dry waste,wet waste}/, a manifest.csv
     covering every input image (kept or dropped, and why), and report.md.

Output folders keep the 'dry waste' / 'wet waste' names, so the existing
train_classifier.py works unchanged:  python train_classifier.py --data_dir <out>

Usage:
  python clean_dataset.py --src ai_dustbin_2 --out cleaned_dataset
"""

import argparse
import csv
import hashlib
import os
import random
import shutil
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASS_FOLDERS = {"dry": "dry waste", "wet": "wet waste"}

# RealWaste class -> dry/wet. 'Miscellaneous Trash' mixes both, so it is left out.
REALWASTE_MAP = {
    "Food Organics": "wet",
    "Vegetation": "wet",
    "Cardboard": "dry",
    "Glass": "dry",
    "Metal": "dry",
    "Paper": "dry",
    "Plastic": "dry",
    "Textile Trash": "dry",
    "Miscellaneous Trash": None,
}


# ─── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Clean and re-split the wet/dry waste dataset")
parser.add_argument("--src",          default="ai_dustbin_2",    help="Folder containing train/ and val/")
parser.add_argument("--out",          default="cleaned_dataset", help="Output folder (recreated each run)")
parser.add_argument("--realwaste",    default=None,              help="Optional path to the RealWaste folder (the one holding 'Cardboard', 'Food Organics', ...)")
parser.add_argument("--ratios",       type=float, nargs=3, default=[0.70, 0.15, 0.15], help="train val test")
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--phash_dist",   type=int,   default=4,  help="Max Hamming distance (of 64 bits) to call two images near-duplicates")
parser.add_argument("--min_side",     type=int,   default=64, help="Drop images smaller than this on either side")
parser.add_argument("--min_std",      type=float, default=4.0, help="Drop near-blank images (grayscale pixel std below this)")
parser.add_argument("--drop_near",    action="store_true",       help="Keep only one image per near-duplicate group instead of grouping them")
args = parser.parse_args()


def long_path(path):
    """Windows without LongPathsEnabled can't open paths over 260 chars; the \\\\?\\ prefix bypasses that."""
    path = os.path.abspath(path)
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + path
    return path


# ─── Perceptual hash (pHash) ──────────────────────────────────────────────────
def _dct_matrix(n):
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    m[0] *= 1 / np.sqrt(2)
    return m * np.sqrt(2 / n)

DCT32 = _dct_matrix(32)

def phash(gray_img):
    """64-bit perceptual hash: low-frequency DCT coefficients vs their median."""
    pixels = np.asarray(gray_img.resize((32, 32), Image.LANCZOS), dtype=np.float64)
    low = (DCT32 @ pixels @ DCT32.T)[:8, :8].flatten()
    bits = low > np.median(low[1:])          # skip the DC term when picking the median
    return int("".join("1" if b else "0" for b in bits), 2)


# ─── Union-find for grouping duplicates ───────────────────────────────────────
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ─── 1. Scan ──────────────────────────────────────────────────────────────────
print(f"[Scan] Looking for images under '{args.src}' ...")
records = []
for split_dir in ("train", "val", "valid", "test"):
    split_path = os.path.join(args.src, split_dir)
    if not os.path.isdir(split_path):
        continue
    for folder in sorted(os.listdir(split_path)):
        label = "dry" if "dry" in folder.lower() else "wet" if "wet" in folder.lower() else None
        folder_path = os.path.join(split_path, folder)
        if label is None or not os.path.isdir(folder_path):
            continue
        for name in sorted(os.listdir(long_path(folder_path))):
            if name.lower().endswith(IMAGE_EXTS):
                records.append({
                    "path": os.path.join(args.src, split_dir, folder, name),
                    "orig_split": split_dir,
                    "label": label,
                    "source": "dataset2" if name.startswith("dataset2_") else "roboflow",
                    "subclass": "",
                    "filename": name,
                })

excluded = []
if args.realwaste:
    for folder in sorted(os.listdir(args.realwaste)):
        folder_path = os.path.join(args.realwaste, folder)
        if not os.path.isdir(folder_path):
            continue
        if folder not in REALWASTE_MAP:
            raise ValueError(f"Unknown RealWaste class folder '{folder}' - add it to REALWASTE_MAP")
        for name in sorted(os.listdir(long_path(folder_path))):
            if not name.lower().endswith(IMAGE_EXTS):
                continue
            rec = {
                "path": os.path.join(folder_path, name),
                "orig_split": "realwaste",
                "label": REALWASTE_MAP[folder],
                "source": "realwaste",
                "subclass": folder,
                "filename": f"realwaste_{folder.replace(' ', '_')}_{name}",
            }
            (records if rec["label"] else excluded).append(rec)
print(f"  Found {len(records)} images" + (f" (+{len(excluded)} RealWaste 'Miscellaneous Trash' left out)" if excluded else ""))

# ─── 2. Read, hash and quality-check every image ──────────────────────────────
print("[Check] Reading images (hashes, size, blank check) ...")
for i, r in enumerate(records):
    full = long_path(r["path"])
    r.update(sha256="", phash=None, width=0, height=0, status="keep", reason="", group="")
    try:
        with open(full, "rb") as f:
            data = f.read()
        r["sha256"] = hashlib.sha256(data).hexdigest()
        with Image.open(full) as img:
            img.load()
            r["width"], r["height"] = img.size
            gray = img.convert("L")
    except Exception as e:
        r["status"], r["reason"] = "drop", f"unreadable ({type(e).__name__})"
        continue

    if min(r["width"], r["height"]) < args.min_side:
        r["status"], r["reason"] = "drop", f"too small ({r['width']}x{r['height']})"
    elif float(np.asarray(gray, dtype=np.float32).std()) < args.min_std:
        r["status"], r["reason"] = "drop", "near-blank image"
    else:
        r["phash"] = phash(gray)

    if (i + 1) % 1000 == 0:
        print(f"  {i + 1}/{len(records)}")

# ─── 3. Exact + near-duplicate grouping ───────────────────────────────────────
print("[Dedup] Grouping exact and near-duplicates ...")
live = [i for i, r in enumerate(records) if r["status"] == "keep"]
uf = UnionFind(len(records))

by_sha = defaultdict(list)
for i in live:
    by_sha[records[i]["sha256"]].append(i)
for idxs in by_sha.values():
    for j in idxs[1:]:
        uf.union(idxs[0], j)

hashes = np.array([records[i]["phash"] for i in live], dtype=np.uint64)
near_pairs = 0
for a in range(len(live)):
    dists = np.bitwise_count(hashes[a + 1:] ^ hashes[a])
    for b in np.nonzero(dists <= args.phash_dist)[0]:
        uf.union(live[a], live[a + 1 + b])
        near_pairs += 1

groups = defaultdict(list)
for i in live:
    groups[uf.find(i)].append(i)

dup_groups = [g for g in groups.values() if len(g) > 1]
conflict_groups = 0
for gid, members in enumerate(dup_groups, start=1):
    exact = len({records[i]["sha256"] for i in members}) == 1
    kind = "exact" if exact else "near"
    for i in members:
        records[i]["group"] = f"{kind}-{gid}"
    if len({records[i]["label"] for i in members}) > 1:
        conflict_groups += 1
        for i in members:
            records[i]["status"], records[i]["reason"] = "drop", f"{kind} duplicate with conflicting labels"
        continue
    # Exact copies: keep one per distinct file. Near-duplicates: keep all unless --drop_near.
    keepers = {}
    for i in sorted(members, key=lambda i: (-records[i]["width"] * records[i]["height"], records[i]["path"])):
        key = "group" if args.drop_near else records[i]["sha256"]
        if key in keepers:
            records[i]["status"] = "drop"
            records[i]["reason"] = f"{'exact' if records[i]['sha256'] == records[keepers[key]]['sha256'] else 'near'} duplicate of {records[keepers[key]]['path']}"
        else:
            keepers[key] = i

print(f"  {len(dup_groups)} duplicate groups ({near_pairs} similar pairs), {conflict_groups} with conflicting labels")

# ─── 4. Stratified split by class + source ────────────────────────────────────
print("[Split] Stratified 70/15/15 by class and source (duplicate groups stay together) ...")
rng = random.Random(args.seed)
SPLITS = ("train", "val", "test")
ratios = [x / sum(args.ratios) for x in args.ratios]

# Units are whole duplicate groups (or single images). A group can span both sources
# (dataset2 re-uses some Roboflow photos), so each unit is bucketed by its first member.
units_by_id = defaultdict(list)
for idx, r in enumerate(records):
    if r["status"] == "keep":
        units_by_id[r["group"] or f"single-{idx}"].append(r)
strata = defaultdict(list)
for unit in units_by_id.values():
    unit.sort(key=lambda r: r["path"])
    strata[(unit[0]["label"], unit[0]["source"])].append(unit)

for key in sorted(strata):
    units = sorted(strata[key], key=lambda u: u[0]["path"])
    rng.shuffle(units)
    units.sort(key=len, reverse=True)          # place big groups first, then fill with singles
    total = sum(len(u) for u in units)
    filled = dict.fromkeys(SPLITS, 0)
    for unit in units:
        # Put the unit in whichever split is furthest below its target size
        split = min(SPLITS, key=lambda s: (filled[s] + len(unit)) / (ratios[SPLITS.index(s)] * total))
        filled[split] += len(unit)
        for r in unit:
            r["split"] = split

# ─── 5. Write output ──────────────────────────────────────────────────────────
if os.path.exists(args.out):
    shutil.rmtree(long_path(args.out))
print(f"[Write] Copying kept images to '{args.out}' ...")
used_names = set()
for r in records:
    r.setdefault("split", "")
    r["new_path"] = ""
    if r["status"] != "keep":
        continue
    dest_dir = os.path.join(args.out, r["split"], CLASS_FOLDERS[r["label"]])
    os.makedirs(long_path(dest_dir), exist_ok=True)
    name, ext = os.path.splitext(r["filename"])
    candidate = r["filename"]
    n = 1
    while (dest_dir, candidate.lower()) in used_names:
        candidate = f"{name}_{n}{ext}"
        n += 1
    used_names.add((dest_dir, candidate.lower()))
    shutil.copy2(long_path(r["path"]), long_path(os.path.join(dest_dir, candidate)))
    r["new_path"] = os.path.join(r["split"], CLASS_FOLDERS[r["label"]], candidate)

fields = ["path", "orig_split", "label", "source", "subclass", "width", "height", "sha256", "phash",
          "group", "status", "reason", "split", "new_path"]
with open(os.path.join(args.out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for r in records:
        row = dict(r)
        row["phash"] = f"{r['phash']:016x}" if r["phash"] is not None else ""
        writer.writerow(row)

# ─── Report ───────────────────────────────────────────────────────────────────
kept = [r for r in records if r["status"] == "keep"]
drop_reasons = Counter(r["reason"].split(" of ")[0] for r in records if r["status"] == "drop")
table = Counter((r["split"], r["label"], r["source"]) for r in kept)
sources = sorted({r["source"] for r in records})
old_split_leaks = sum(
    1 for g in dup_groups if {"train", "val"} <= {records[i]["orig_split"] for i in g}
)
cross_source = Counter(
    " + ".join(sorted({records[i]["source"] for i in g})) for g in dup_groups
    if len({records[i]["source"] for i in g}) > 1
)

lines = [
    "# Dataset cleaning report", "",
    f"- Sources: {', '.join(sources)}",
    f"- Input images: **{len(records)}**" + (f" (RealWaste 'Miscellaneous Trash' left out: {len(excluded)})" if excluded else ""),
    f"- Kept: **{len(kept)}**  |  Dropped: **{len(records) - len(kept)}**",
    f"- Duplicate groups: {len(dup_groups)} (of which {old_split_leaks} had copies in both the old ai_dustbin_2 train and val folders)",
    f"- Near-duplicate threshold: pHash Hamming distance <= {args.phash_dist} / 64",
    f"- Near-duplicates: {'one kept per group' if args.drop_near else 'all kept, each group confined to one split'}",
    f"- Duplicate groups spanning sources: {dict(cross_source) or 'none'}",
    f"- Duplicate groups spanning more than one new split: {sum(1 for g in dup_groups if len({records[i]['split'] for i in g if records[i]['status'] == 'keep'}) > 1)}",
    f"- Split ratios: {args.ratios}, seed {args.seed}", "",
    "## Dropped, by reason", "", "| Reason | Count |", "|---|---|",
]
lines += [f"| {k} | {v} |" for k, v in drop_reasons.most_common()]
cols = [(lab, src) for lab in ("dry", "wet") for src in sources]
lines += ["", "## Kept images per split", "",
          "| Split | " + " | ".join(f"{lab.title()} ({src})" for lab, src in cols) + " | Total | Wet share |",
          "|---" * (len(cols) + 3) + "|"]
for split in ("train", "val", "test"):
    c = [table[(split, lab, src)] for lab, src in cols]
    total, wet = sum(c), sum(n for (lab, _), n in zip(cols, c) if lab == "wet")
    lines.append(f"| {split} | " + " | ".join(map(str, c)) + f" | {total} | {wet / max(total, 1):.1%} |")

lines += ["", "## Wet share within each source (kept images)", "", "| Source | Dry | Wet | Wet share |", "|---|---|---|---|"]
for src in sources:
    d, w = sum(1 for r in kept if r["source"] == src and r["label"] == "dry"), sum(1 for r in kept if r["source"] == src and r["label"] == "wet")
    lines.append(f"| {src} | {d} | {w} | {w / max(d + w, 1):.1%} |")

train_counts = Counter(r["label"] for r in kept if r["split"] == "train")
n = sum(train_counts.values())
lines += ["", "## Class weights for training (as train_classifier.py computes them)", "",
          f"- dry: {n / (2 * train_counts['dry']):.3f}",
          f"- wet: {n / (2 * train_counts['wet']):.3f}", "",
          "Every split has the same source mix, so per-source results can be reported from `manifest.csv`.", ""]

with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"[Done] Wrote {args.out}/manifest.csv and {args.out}/report.md")
