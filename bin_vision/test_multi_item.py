"""
test_multi_item.py
------------------
Tests the full multi_item.py pipeline (detector + classifier) on synthetic
plate scenes (see synth.py) until real bin photos exist.

Scenarios: single dry/wet, two of the same class, mixed pair (separate),
mixed pair (touching), three items with at least one of each class.

Reports two things separately:
  - Counting: did the detector find the right number of blobs?
  - Decision: did the full pipeline (detector + classifier) choose correctly?
    "Unsafe" = it tilted a mixed load, or tilted a single-class load the wrong way.

This uses the class-agreement policy (tilt when all items agree). For the
one-item-at-a-time warning, see test_counting.py.

Usage:
  python test_multi_item.py --clean ../../cleaned_v2 --masks ../../dataset_D_v2/masks \
      --classifier_dir ../../ai_dustbin_2 --n 60 --save_examples examples/
"""

import argparse
import os
import random
from collections import Counter, defaultdict

import cv2
import numpy as np

from multi_item import ItemDetector, classify_items, decide, draw, load_classifier
from synth import ItemPool, SceneMaker

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

parser = argparse.ArgumentParser()
parser.add_argument("--clean",          required=True, help="cleaned dataset folder (original images + manifest.csv)")
parser.add_argument("--masks",          required=True, help="dataset D masks folder for the same dataset")
parser.add_argument("--classifier_dir", required=True)
parser.add_argument("--model_path",     default=None)
parser.add_argument("--n",              type=int, default=60, help="Scenes per scenario")
parser.add_argument("--threshold",      type=float, default=0.6)
parser.add_argument("--seed",           type=int, default=0)
parser.add_argument("--save_examples",  default=None, help="Folder for a few annotated scenes")
parser.add_argument("--split",          action="store_true", help="Turn on watershed splitting of touching items")
args = parser.parse_args()

rng = random.Random(args.seed)
pool = ItemPool(args.clean, args.masks, "test", seed=args.seed)
maker = SceneMaker(pool, seed=args.seed)
print(f"Item pool (test split with masks): {pool.sizes()}")

classifier = load_classifier(args.classifier_dir, args.model_path)
if args.save_examples:
    os.makedirs(args.save_examples, exist_ok=True)

results = defaultdict(Counter)
count_ok = Counter()
for name, (labels, touching) in SCENARIOS.items():
    for k in range(args.n):
        labels_k = [l or rng.choice(["dry", "wet"]) for l in labels]
        reference, frame = maker.make_scene(labels_k, touching=touching)

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
