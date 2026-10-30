import csv, glob, os, sys
import cv2, numpy as np
repo, piles, bgdir = sys.argv[1:4]
sys.path.insert(0, os.path.join(repo, "bin_vision")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg import load_refs, segment
names = sorted(glob.glob(os.path.join(bgdir, "*.jpg")))
rows = list(csv.DictReader(open(os.path.join(piles, "items.csv"), encoding="utf-8")))
frames = [cv2.imread(os.path.join(piles, r["file"])) for r in rows]
for th in (14, 16, 18, 20, 24, 28):
    refs = load_refs(bgdir, th)
    # empty plate vs each other empty plate (not itself): area wrongly called an item
    fa = []
    for f in names:
        others = {k: v for k, v in refs.items() if k != os.path.basename(f)}
        it, *_ = segment(others, cv2.imread(f)); fa.append(it.area_frac if it else 0)
    none = 0; areas = []
    for f in frames:
        it, *_ = segment(refs, f); none += it is None; areas.append(it.area_frac if it else 0)
    print(f"th {th}: no item {none:3d} | empty plate flagged: median {np.median(fa):.4f} max {max(fa):.4f} | item area median {np.median(areas):.3f}")
