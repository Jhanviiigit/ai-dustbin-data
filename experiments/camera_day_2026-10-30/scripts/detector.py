"""Detector numbers on the single-item photos: area fraction, and how often the SEPARATE rule
wrongly stops a single item at max_area_frac 0.09 (now) and other values."""
import csv, os, sys
import cv2, numpy as np
repo, piles, bgdir, cdir = sys.argv[1:5]
sys.path.insert(0, os.path.join(repo, "bin_vision")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg import load_refs, segment
from multi_item import ShapeRule, blob_features, suspect_multiple, DEFAULT_RULE
refs = load_refs(bgdir, 16)
rows = [r for r in csv.DictReader(open(os.path.join(cdir, "crops.csv"), encoding="utf-8")) if r["crop"]]
feats = []
for r in rows:
    f = cv2.imread(os.path.join(piles, r["file"])); it, *_ = segment(refs, f, r["ref"])
    feats.append(blob_features(f, it))
area = np.array([x["area_frac"] for x in feats])
print(f"single-item photos {len(area)}; area_frac median {np.median(area):.3f}  p95 {np.percentile(area, 95):.3f}  max {area.max():.3f}")
print("feature keys:", list(feats[0]))
for maf in (0.09, 0.15, 0.25, 0.30, 0.40, 0.50):
    rule = ShapeRule(max_area_frac=maf, min_notch2=DEFAULT_RULE.min_notch2)
    stops = [bool(suspect_multiple(x, rule)) for x in feats]
    by_area = int((area > maf).sum())
    print(f"max_area_frac {maf:.2f}: false SEPARATE {sum(stops)}/{len(stops)} = {sum(stops)/len(stops):.1%}  (area {by_area}, notch {sum(stops)-by_area if sum(stops)>=by_area else '?'})")
