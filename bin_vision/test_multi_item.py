"""
test_multi_item.py
------------------
Tests multi_item.py on synthetic plate scenes until real bin photos exist.

Each scene is a fake empty plate (flat colour + texture + lighting gradient)
with real waste cut-outs pasted on, using the masks from dataset D. The
reference frame and the scene get independent sensor noise and a small
exposure change, and items cast soft shadows, so the background subtraction
is not tested on a perfect copy.

Scenarios: single dry/wet, two of the same class, mixed pair (separate),
mixed pair (touching), three items with at least one of each class.

Reports two things separately:
  - Counting: did the detector find the right number of blobs?
  - Decision: did the full pipeline (detector + classifier) choose correctly?
    "Unsafe" = it tilted a mixed load, or tilted a single-class load the wrong way.

Usage:
  python test_multi_item.py --clean ../../cleaned_dataset --masks ../../dataset_D/masks \
      --classifier_dir ../../ai_dustbin_2 --n 60 --save_examples examples/
"""

import argparse
import csv
import os
import random
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image, ImageOps

from multi_item import ItemDetector, classify_items, decide, draw, load_classifier

SCENARIOS = {
    # name: (labels, touching)
    "single_dry":     (["dry"], False),
    "single_wet":     (["wet"], False),
    "two_dry":        (["dry", "dry"], False),
    "two_wet":        (["wet", "wet"], False),
    "mixed_apart":    (["dry", "wet"], False),
    "mixed_touching": (["dry", "wet"], True),
    "three_mixed":    (["dry", "wet", None], False),   # None = random class
}
PLATES = [(150, 150, 146), (70, 72, 75), (170, 120, 60)]   # light grey, dark grey, blue (BGR)
W, H = 640, 480

parser = argparse.ArgumentParser()
parser.add_argument("--clean",          required=True, help="cleaned_dataset folder (original images + manifest.csv)")
parser.add_argument("--masks",          required=True, help="dataset_D/masks folder")
parser.add_argument("--classifier_dir", required=True)
parser.add_argument("--model_path",     default=None)
parser.add_argument("--n",              type=int, default=60, help="Scenes per scenario")
parser.add_argument("--threshold",      type=float, default=0.6)
parser.add_argument("--seed",           type=int, default=0)
parser.add_argument("--save_examples",  default=None, help="Folder for a few annotated scenes")
parser.add_argument("--split",          action="store_true", help="Turn on watershed splitting of touching items")
args = parser.parse_args()

rng = random.Random(args.seed)
np_rng = np.random.default_rng(args.seed)


def long_path(path):
    path = os.path.abspath(path)
    return "\\\\?\\" + path if os.name == "nt" and not path.startswith("\\\\?\\") else path


# ─── Item pool: test-split images that already have a mask ───────────────────
pool = defaultdict(list)
for r in csv.DictReader(open(os.path.join(args.clean, "manifest.csv"), encoding="utf-8")):
    if r["status"] != "keep" or r["split"] != "test":
        continue
    mask_path = os.path.join(args.masks, os.path.splitext(r["new_path"])[0] + ".png")
    if os.path.exists(long_path(mask_path)):
        pool[r["label"]].append((os.path.join(args.clean, r["new_path"]), mask_path))
print(f"Item pool (test split with masks): dry={len(pool['dry'])} wet={len(pool['wet'])}")

cutout_cache = {}


def load_cutout(img_path, mask_path):
    """Return (rgb, alpha) cropped to the object, or None if the mask is unusable."""
    key = img_path
    if key not in cutout_cache:
        img = ImageOps.exif_transpose(Image.open(long_path(img_path))).convert("RGB")
        img.thumbnail((512, 512))
        mask = Image.open(long_path(mask_path)).convert("L")
        if mask.size != img.size:
            mask = mask.resize(img.size)
        a = np.asarray(mask)
        cover = (a > 127).mean()
        ys, xs = np.nonzero(a > 127)
        # Skip failed cut-outs and ones that kept most of the background
        if not (0.05 <= cover <= 0.6) or len(xs) == 0:
            cutout_cache[key] = None
        else:
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            rgb = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)[y0:y1, x0:x1]
            cutout_cache[key] = (rgb, a[y0:y1, x0:x1].astype(np.float32) / 255)
    return cutout_cache[key]


def pick_cutout(label):
    while True:
        c = load_cutout(*rng.choice(pool[label]))
        if c is not None:
            return c


def make_plate(colour):
    base = np.full((H, W, 3), colour, np.float32)
    texture = cv2.GaussianBlur(np_rng.normal(0, 6, (H, W)).astype(np.float32), (0, 0), 3)[..., None]
    gx = np.linspace(-1, 1, W)[None, :, None]
    gy = np.linspace(-1, 1, H)[:, None, None]
    lighting = 1 + 0.12 * (rng.uniform(-1, 1) * gx + rng.uniform(-1, 1) * gy)
    return base * lighting + texture


def snap(plate, exposure=1.0):
    """Camera capture: exposure change + per-frame sensor noise."""
    return np.clip(plate * exposure + np_rng.normal(0, 3, plate.shape), 0, 255).astype(np.uint8)


def paste(scene, cutout, cx, cy, size):
    rgb, alpha = cutout
    scale = size / max(rgb.shape[:2])
    rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
    alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]))[..., None]
    h, w = rgb.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    xs0, ys0, xs1, ys1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xs1 <= xs0 or ys1 <= ys0:
        return None
    sub_rgb = rgb[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    sub_a = alpha[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    # Soft shadow, offset down-right
    shadow = cv2.GaussianBlur(sub_a[..., 0], (0, 0), 6)[..., None] * 0.35
    sx, sy = 8, 8
    region = scene[min(H, ys0 + sy):min(H, ys1 + sy), min(W, xs0 + sx):min(W, xs1 + sx)]
    region *= 1 - shadow[:region.shape[0], :region.shape[1]]
    scene[ys0:ys1, xs0:xs1] = scene[ys0:ys1, xs0:xs1] * (1 - sub_a) + sub_rgb * sub_a
    return (xs0, ys0, xs1, ys1)


def place(n_items, touching):
    """Centres and sizes for n items: spread apart, or the first two touching."""
    sizes = [rng.randint(140, 200) for _ in range(n_items)]
    if touching:
        cx, cy = rng.randint(220, 420), rng.randint(180, 300)
        gap = int((sizes[0] + sizes[1]) * 0.33)          # bounding boxes overlap -> objects meet
        return [(cx - gap // 2, cy, sizes[0]), (cx + gap // 2, cy + rng.randint(-20, 20), sizes[1])]
    slots = [(160, 140), (480, 140), (160, 340), (480, 340), (320, 240)]
    rng.shuffle(slots)
    return [(x + rng.randint(-25, 25), y + rng.randint(-25, 25), s) for (x, y), s in zip(slots, sizes)]


classifier = load_classifier(args.classifier_dir, args.model_path)
if args.save_examples:
    os.makedirs(args.save_examples, exist_ok=True)

results = defaultdict(Counter)
count_ok = Counter()
for name, (labels, touching) in SCENARIOS.items():
    for k in range(args.n):
        labels_k = [l or rng.choice(["dry", "wet"]) for l in labels]
        plate = make_plate(rng.choice(PLATES))
        reference = snap(plate)
        scene = plate.copy()
        for (cx, cy, size), label in zip(place(len(labels_k), touching), labels_k):
            paste(scene, pick_cutout(label), cx, cy, size)
        frame = snap(scene, exposure=rng.uniform(0.93, 1.07))

        detector = ItemDetector(split_touching=args.split)
        detector.set_reference(reference)
        items = detector.detect(frame)
        decision = decide(classify_items(frame, items, classifier), args.threshold)

        expected_blobs = 1 if touching else len(labels_k)
        count_ok[name] += len(items) == expected_blobs
        truth = set(labels_k)
        if len(truth) > 1:
            outcome = "correct" if decision.action == "MIXED" else \
                      "safe (asked human)" if decision.action == "ASK_HUMAN" else \
                      "UNSAFE tilt" if decision.action.startswith("TILT") else decision.action
        else:
            want = f"TILT:{truth.pop().upper()}"
            outcome = "correct" if decision.action == want else \
                      "safe (asked human)" if decision.action in ("ASK_HUMAN", "MIXED") else \
                      "UNSAFE tilt" if decision.action.startswith("TILT") else decision.action
        results[name][outcome] += 1

        if args.save_examples and k < 2:
            cv2.imwrite(os.path.join(args.save_examples, f"{name}_{k}.jpg"), np.hstack([reference, draw(frame, decision)]))

print(f"\nThreshold {args.threshold}, {args.n} scenes per scenario\n")
print(f"| Scenario | Blob count right | Correct | Asked human | Unsafe tilt | Other |")
print(f"|---|---|---|---|---|---|")
for name in SCENARIOS:
    r = results[name]
    other = sum(v for k, v in r.items() if k not in ("correct", "safe (asked human)", "UNSAFE tilt"))
    print(f"| {name} | {count_ok[name] / args.n:.0%} | {r['correct'] / args.n:.0%} | "
          f"{r['safe (asked human)'] / args.n:.0%} | {r['UNSAFE tilt'] / args.n:.0%} | {other / args.n:.0%} |")
