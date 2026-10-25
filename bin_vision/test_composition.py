"""
test_composition.py
-------------------
Tests whether the bin tilts only when every item on the plate belongs in the
same compartment (dry+dry, wet+wet) and stops for dry+wet.

Runs the real detector (blobs, drop counter, shape checks, assess()) on
synthetic frame sequences (synth.py). The student model is replaced by a
STAND-IN that reads the true label of the pixels in each crop from the scene,
then makes mistakes at a set rate. This measures the detector's logic on its
own; re-run with the real student once it exists.

Stand-in behaviour per crop:
  - label = the majority true label inside the item's mask; a merged dry+wet
    blob gets the majority label with HIGH confidence (the worst case, and
    what the old model did on merged blobs)
  - with probability --unsure: confidence below the threshold
  - with probability --error: the wrong label, confidently

Outcomes:
  correct   tilted the right way, or MIXED on a dry+wet load
  stopped   did not tilt (ASK_HUMAN, SEPARATE, or MIXED on a same-class load): safe but unneeded
  UNSAFE    tilted a dry+wet load, or tilted a same-class load the wrong way

Usage:
  python test_composition.py --clean ../../cleaned_v2 --masks ../../dataset_D_v2/masks --n 80
"""

import argparse
import random

import numpy as np

from multi_item import DEFAULT_RULE, DropCounter, ItemDetector, assess
from synth import ItemPool, SceneMaker, TRUTH_IDS

# name: (labels, touching, dropped together); None = random class
SCENARIOS = {
    "dry":                          (["dry"], False, True),
    "wet":                          (["wet"], False, True),
    "dry+dry apart":                (["dry", "dry"], False, False),
    "wet+wet apart":                (["wet", "wet"], False, False),
    "dry+wet apart":                (["dry", "wet"], False, False),
    "dry+dry touching, one by one": (["dry", "dry"], True, False),
    "dry+wet touching, one by one": (["dry", "wet"], True, False),
    "dry+dry touching, together":   (["dry", "dry"], True, True),
    "dry+wet touching, together":   (["dry", "wet"], True, True),
    "three, mixed, apart":          (["dry", "wet", None], False, False),
}

parser = argparse.ArgumentParser()
parser.add_argument("--clean",     required=True, help="cleaned dataset folder (images + manifest.csv)")
parser.add_argument("--masks",     required=True, help="dataset D masks folder for the same dataset")
parser.add_argument("--n",         type=int, default=80, help="Scenes per scenario")
parser.add_argument("--threshold", type=float, default=0.6)
parser.add_argument("--unsure",    type=float, default=0.05, help="Stand-in: share of crops below the threshold")
parser.add_argument("--error",     type=float, default=0.05, help="Stand-in: share of crops confidently wrong")
parser.add_argument("--seed",      type=int, default=3)
parser.add_argument("--out",       default=None, help="Optional markdown file for the results")
args = parser.parse_args()

ID_LABEL = {v: k for k, v in TRUTH_IDS.items()}
CONFIGS = {"perfect stand-in": (0.0, 0.0), f"stand-in, {args.error:.0%} wrong + {args.unsure:.0%} unsure": (args.error, args.unsure)}


def make_stand_in(truth_ref, error, unsure, rng):
    """classify(frame, item) reading the truth map currently in truth_ref[0]."""
    def classify(frame, item):
        ids = truth_ref[0][item.mask]
        counts = {lab: int((ids == i).sum()) for lab, i in TRUTH_IDS.items()}
        if sum(counts.values()) == 0:
            return rng.choice(["dry", "wet"]), 0.55          # mask covers only shadow/plate
        label = max(counts, key=counts.get)
        r = rng.random()
        if r < error:
            return ("wet" if label == "dry" else "dry"), rng.uniform(0.65, 0.95)
        if r < error + unsure:
            return label, rng.uniform(0.5, args.threshold - 0.01)
        return label, rng.uniform(0.8, 0.99)
    return classify


def load_type(labels):
    """'all dry', 'all wet' or 'mixed': what decides the tilt, regardless of item count."""
    kinds = set(labels)
    return "mixed" if len(kinds) > 1 else f"all {kinds.pop()}"


def outcome(action, truth_labels):
    mixed = len(set(truth_labels)) > 1
    if mixed:
        if action == "MIXED":
            return "correct"
        return "UNSAFE" if action.startswith("TILT") else "stopped"
    want = f"TILT:{truth_labels[0].upper()}"
    if action == want:
        return "correct"
    return "UNSAFE" if action.startswith("TILT") else "stopped"


pool = ItemPool(args.clean, args.masks, "test", seed=args.seed)
maker = SceneMaker(pool, seed=args.seed)
rng = random.Random(args.seed)
print(f"Item pool (test split): {pool.sizes()}")

results = {c: {s: {"correct": 0, "stopped": 0, "UNSAFE": 0, "comp": 0} for s in SCENARIOS} for c in CONFIGS}
for name, (labels, touching, together) in SCENARIOS.items():
    for _ in range(args.n):
        labels_k = [l or rng.choice(["dry", "wet"]) for l in labels]
        truth_type = load_type(labels_k)
        reference, frames, truths = maker.make_sequence(labels_k, touching=touching, together=together)
        detector = ItemDetector()
        detector.set_reference(reference)
        truth_ref = [None]
        runs = {}
        for cfg, (error, unsure) in CONFIGS.items():
            runs[cfg] = (DropCounter(detector), make_stand_in(truth_ref, error, unsure, random.Random(rng.random())))
        for frame, truth in zip(frames, truths):
            truth_ref[0] = truth
            for counter, classify in runs.values():
                counter.update(frame, classify)
        final = frames[-1]
        items = detector.detect(final)
        for cfg, (counter, classify) in runs.items():
            d = assess(final, items, classify, args.threshold, DEFAULT_RULE, counter.drop_items)
            res = results[cfg][name]
            res[outcome(d.action, labels_k)] += 1
            res["comp"] += bool(d.composition) and load_type(d.composition.split("+")) == truth_type
    print(f"  done: {name}")

lines = [f"Test-split items, {args.n} scenes per scenario, threshold {args.threshold}. "
         f"Stand-in classifier instead of the student model (see test_composition.py).", ""]
for cfg in CONFIGS:
    lines += [f"**{cfg}**", "", "| Load | Correct | Stopped (safe) | Unsafe tilt | Load type right (all dry / all wet / mixed) |",
              "|---|---|---|---|---|"]
    for name in SCENARIOS:
        r = results[cfg][name]
        lines.append(f"| {name} | {r['correct'] / args.n:.0%} | {r['stopped'] / args.n:.0%} | "
                     f"{r['UNSAFE'] / args.n:.0%} | {r['comp'] / args.n:.0%} |")
    lines.append("")
print("\n" + "\n".join(lines))
if args.out:
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
