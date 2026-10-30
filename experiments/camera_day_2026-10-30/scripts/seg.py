"""Segment the one item in a pile photo with the bin's detector (empty-plate subtraction).
Changes from the live bin, for one-item photos only:
  - the empty-plate reference is the one whose plate border matches this photo best (several were taken);
  - thin strips touching the left/right frame edge are dropped (the plate edge shifts between shots);
  - the item is the largest remaining blob plus fragments lying within 8% of the frame width of it."""
import glob, os, sys
import cv2, numpy as np
from multi_item import ItemDetector, Item

def load_refs(bgdir, th):
    out = {}
    for f in sorted(glob.glob(os.path.join(bgdir, "*.jpg"))):
        d = ItemDetector(diff_threshold=th); d.set_reference(cv2.imread(f)); out[os.path.basename(f)] = d
    return out

def _border(a):
    h, w = a.shape[:2]; b = max(4, h // 8)
    return np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3), a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])

def pick_ref(refs, frame):
    lab = next(iter(refs.values()))._prep(frame)
    def cost(d):
        diff = lab - d.reference; diff = diff - np.median(diff, axis=(0, 1))
        return float(np.median(np.abs(_border(diff)).sum(1)))
    return min(refs, key=lambda n: cost(refs[n]))

def edge_strip(it, w, h, frac=0.12):
    x, y, bw, bh = it.box
    return (x <= 1 or x + bw >= w - 1) and bw < frac * w

def segment(refs, frame, ref=None):
    """-> (Item or None, ref name, blobs found, blobs dropped as edge strips)"""
    ref = ref or pick_ref(refs, frame)
    h, w = frame.shape[:2]
    items = refs[ref].detect(frame)
    keep = [i for i in items if not edge_strip(i, w, h)]
    if not keep:
        return None, ref, len(items), len(items)
    main = max(keep, key=lambda i: i.area_frac); gap = 0.08 * w
    x, y, bw, bh = main.box
    def near(i):
        a, b, c, d = i.box
        return a <= x + bw + gap and a + c >= x - gap and b <= y + bh + gap and b + d >= y - gap
    m = np.zeros((h, w), bool)
    for i in keep:
        if i is main or near(i): m |= i.mask
    ys, xs = np.nonzero(m)
    box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    return Item(box=box, area_frac=float(m.mean()), mask=m), ref, len(items), len(items) - len(keep)
