"""Regroup the Organik crops by image similarity and re-split them (dataset_organik_v2).

split_organik.py groups photos by perceptual hash and file order. That misses the same item
photographed in another pose, so such photos can land in different splits and the scores are
too good. This step fixes it: images whose ImageNet features are alike are put in one group, and
whole groups go to train, val or test.

Input: the folder split_organik.py wrote (its manifest.csv lists every kept image; its own
train/val/test folders are ignored). Output: the same images in a new split + manifest.csv and
report.md.

  python data_prep/split_organik.py   --src grey/organik --out dataset_organik
  python data_prep/regroup_organik.py --src dataset_organik --out dataset_organik_v2

How:
  1. Features: MobileNetV2 (ImageNet weights), global-average-pooled, L2-normalised, at 224 px.
  2. Clusters: average-linkage on cosine distance, cut at --dist, run inside each class (a wet and
     a dry item are never grouped). The earlier hash / neighbour groups are kept whole.
  3. Split: every cluster goes whole into test, val or train. val and test hold the same number of
     images per class (--eval_frac of the smaller class each); everything else is train. Images over
     the quota are discarded, never moved to train.
  4. Check: the nearest train image of every val/test image is reported (cosine similarity of the
     same features). With the default --dist no val/test image has a train image at >= 0.90.

Same seed, same input, same output.
"""
import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from torchvision import models, transforms

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--src", required=True, help="folder written by split_organik.py (train/val/test + manifest.csv)")
parser.add_argument("--out", required=True, help="output folder")
parser.add_argument("--dist", type=float, default=0.18, help="cosine distance at which clusters are cut (0.18 = similarity 0.82)")
parser.add_argument("--eval_frac", type=float, default=0.15, help="val and test size, as a share of the smaller class, each")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

CLASSES = ["dry waste", "wet waste"]
src, out = Path(args.src), Path(args.out)
if out.exists():
    raise SystemExit(f"{out} exists; remove it or pick another --out")

man = list(csv.DictReader(open(src / "manifest.csv", newline="", encoding="utf-8")))
y = np.array([CLASSES.index(r["class_folder"]) for r in man])
old = np.array([r["group"] for r in man])

# 1. features
net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).eval()
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
feats = []
with torch.no_grad():
    for i in range(0, len(man), 64):
        batch = [tf(Image.open(src / r["split"] / r["class_folder"] / r["filename"]).convert("RGB")) for r in man[i:i + 64]]
        feats.append(torch.nn.functional.normalize(net.features(torch.stack(batch)).mean((2, 3)), dim=1))
F = torch.cat(feats).numpy()
S = F @ F.T

# 2. clusters inside each class, then keep the earlier groups whole
lab = np.zeros(len(man), dtype=int)
nxt = 1
for c in (0, 1):
    idx = np.nonzero(y == c)[0]
    D = np.clip(1 - S[np.ix_(idx, idx)], 0, None)
    np.fill_diagonal(D, 0)
    cl = fcluster(linkage(squareform(D, checks=False), "average"), args.dist, "distance")
    lab[idx] = cl + nxt
    nxt += cl.max() + 1
parent = {g: g for g in set(lab)}


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


first = {}
for g_old, g_new in zip(old, lab):
    if g_old in first:
        parent[find(g_new)] = find(first[g_old])
    else:
        first[g_old] = g_new
lab = np.array([find(g) for g in lab])
gid = {g: n for n, g in enumerate(sorted(set(lab), key=lambda g: int(np.nonzero(lab == g)[0][0])))}
lab = np.array([gid[g] for g in lab])
sizes = np.bincount(lab)
print(f"{len(sizes)} groups (was {len(set(old))}); largest {sizes.max()}, singles {(sizes == 1).sum()}")

# 3. split by whole groups; val and test equal per class
rng = random.Random(args.seed)
quota = round(args.eval_frac * np.bincount(y).min())
split = np.array(["train"] * len(man), dtype=object)
for c in (0, 1):
    groups = sorted({g for g in lab[y == c]})
    rng.shuffle(groups)
    for s in ("test", "val"):
        filled = 0
        for g in groups:
            m = np.nonzero(lab == g)[0]
            if split[m[0]] == "train" and filled + len(m) <= quota:
                split[m] = s
                filled += len(m)
drop = set()
for s in ("val", "test"):
    cnt = [int(((split == s) & (y == c)).sum()) for c in (0, 1)]
    for c in (0, 1):
        extra = cnt[c] - min(cnt)
        if extra:
            idx = list(np.nonzero((split == s) & (y == c))[0])
            rng.shuffle(idx)
            drop.update(idx[:extra])
keep = [i for i in range(len(man)) if i not in drop]

# 4. leakage check: nearest train image of every val / test image
tr = np.array([i for i in keep if split[i] == "train"])
for s in ("val", "test"):
    ev = np.array([i for i in keep if split[i] == s])
    nn = S[np.ix_(ev, tr)].max(1)
    print(f"{s}: {len(ev)} images, nearest train image similarity: median {np.median(nn):.3f}, "
          f">=0.90: {(nn >= 0.90).sum()}, >=0.93: {(nn >= 0.93).sum()}")

out_rows = []
for i in keep:
    r = man[i]
    dest = out / split[i] / r["class_folder"]
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / r["split"] / r["class_folder"] / r["filename"], dest / r["filename"])
    out_rows.append({"split": split[i], "class_folder": r["class_folder"], "filename": r["filename"],
                     "new_path": f"{split[i]}/{r['class_folder']}/{r['filename']}", "source": "organik",
                     "group": int(lab[i]), "group_v1": r["group"], "source_folder": r["source_folder"]})
with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
    w.writeheader()
    w.writerows(out_rows)

counts = Counter((r["split"], r["class_folder"]) for r in out_rows)
lines = ["# Organik split v2 (embedding clusters)", "",
         f"- {len(man)} images regrouped by MobileNetV2 (ImageNet) feature similarity: average linkage, cosine distance <= {args.dist}, "
         f"within each class; the earlier hash / neighbour groups kept whole",
         f"- {len(sizes)} groups, largest {sizes.max()}; seed {args.seed}; val and test {quota} images per class each; "
         f"discarded to equalise: {len(drop)}", "",
         "| Split | dry waste | wet waste | Total |", "|---|---|---|---|"]
for s in ("train", "val", "test"):
    d, w_ = counts[(s, "dry waste")], counts[(s, "wet waste")]
    lines.append(f"| {s} | {d} | {w_} | {d + w_} |")
(out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
