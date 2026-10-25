"""
test_counting.py
----------------
Measures the one-item-at-a-time warning on synthetic scenes. No classifier
is involved: the warning only needs to count items.

Two numbers per method:
  - warning recall: share of multi-item loads that get the MULTIPLE warning
  - false alarms:   share of single items that wrongly get it

Methods, each adding to the one before:
  count       2+ blobs
  + shape     a single blob that looks like several items (ShapeRule)
  + drops     2+ separate drops seen in the frame sequence (DropCounter)

The ShapeRule thresholds are tuned on scenes built from VAL-split items
(keeping false alarms on single items at or below --max_false_alarm), then
every method is measured on fresh scenes built from TEST-split items.

Usage:
  python test_counting.py --clean ../../cleaned_v2 --masks ../../dataset_D_v2/masks --n 150
"""

import argparse
import itertools
import random

import numpy as np

from multi_item import DropCounter, ItemDetector, ShapeRule, blob_features, check_count, suspect_multiple
from synth import ItemPool, SceneMaker

# name: (number of items, touching, dropped together)
SCENARIOS = {
    "single":                     (1, False, True),
    "two apart":                  (2, False, False),
    "two touching, one by one":   (2, True, False),
    "two touching, together":     (2, True, True),
    "three apart":                (3, False, False),
}

parser = argparse.ArgumentParser()
parser.add_argument("--clean",            required=True, help="cleaned dataset folder (images + manifest.csv)")
parser.add_argument("--masks",            required=True, help="dataset D masks folder for the same dataset")
parser.add_argument("--n",                type=int, default=150, help="Scenes per scenario, for tuning and again for testing")
parser.add_argument("--max_false_alarm",  type=float, default=0.05)
parser.add_argument("--out",              default=None, help="Optional markdown file for the results")
args = parser.parse_args()


def run_scenes(split, seed):
    """Detector outputs for n scenes per scenario: blob count, drops, single-blob features."""
    pool = ItemPool(args.clean, args.masks, split, seed=seed)
    maker = SceneMaker(pool, seed=seed)
    rng = random.Random(seed)
    print(f"[{split}] item pool: {pool.sizes()}")
    out = {}
    for name, (n_items, touching, together) in SCENARIOS.items():
        rows = []
        for _ in range(args.n):
            labels = [rng.choice(["dry", "wet"]) for _ in range(n_items)]
            reference, frames = maker.make_sequence(labels, touching=touching, together=together)
            detector = ItemDetector()
            detector.set_reference(reference)
            counter = DropCounter(detector)
            for f in frames:
                counter.update(f)
            final = frames[-1]
            items = detector.detect(final)
            rows.append({
                "frame": final,
                "items": items,
                "drops": counter.drops,
                "features": blob_features(final, items[0]) if len(items) == 1 else None,
            })
        out[name] = rows
    return out


def warn_rates(results, rule, use_drops):
    rates = {}
    for name, rows in results.items():
        warned = 0
        for r in rows:
            n_blobs = len(r["items"])
            if n_blobs > 1:
                warned += 1
            elif use_drops and r["drops"] > 1:
                warned += 1
            elif rule is not None and r["features"] is not None and suspect_multiple(r["features"], rule):
                warned += 1
        rates[name] = warned / len(rows)
    return rates


# ─── Tune the shape rule on val-split scenes ─────────────────────────────────
tune = run_scenes("val", seed=1)
grid = itertools.product([0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
                         [1.0, 0.25, 0.2, 0.17, 0.15, 0.13, 0.11, 0.09],
                         [float("inf"), 5.0, 4.0, 3.0],
                         [float("inf"), 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25])
best, best_key = None, None
for solidity, area, sep, notch in grid:
    rule = ShapeRule(min_solidity=solidity, max_area_frac=area, min_color_sep=sep, min_notch2=notch)
    r = warn_rates(tune, rule, use_drops=False)
    if r["single"] > args.max_false_alarm:
        continue
    key = (r["two touching, together"], -r["single"])
    if best_key is None or key > best_key:
        best, best_key = rule, key
print(f"\nTuned on val: {best}  (touching-together recall {best_key[0]:.0%}, single false alarm {-best_key[1]:.0%})")

# ─── Measure on test-split scenes ────────────────────────────────────────────
test = run_scenes("test", seed=2)
methods = {
    "count only":              warn_rates(test, None, use_drops=False),
    "count + shape":           warn_rates(test, best, use_drops=False),
    "count + drops":           warn_rates(test, None, use_drops=True),
    "count + shape + drops":   warn_rates(test, best, use_drops=True),
}

# Which shape rule fired, on the test scenes where it decided the outcome
fired = {}
for name in SCENARIOS:
    reasons = [suspect_multiple(r["features"], best).split(" (")[0]
               for r in test[name] if r["features"] is not None and suspect_multiple(r["features"], best)]
    fired[name] = {k: reasons.count(k) for k in sorted(set(reasons))}

multi = [s for s in SCENARIOS if s != "single"]
lines = [f"Test scenes: {args.n} per scenario, test-split items. Shape rule tuned on val-split scenes: "
         f"min_solidity={best.min_solidity}, max_area_frac={best.max_area_frac}, min_color_sep={best.min_color_sep}, "
         f"min_notch2={best.min_notch2}.", "",
         "| Method | False alarm (single) | " + " | ".join(f"Recall: {s}" for s in multi) + " |",
         "|---" * (len(multi) + 2) + "|"]
for m, r in methods.items():
    lines.append(f"| {m} | {r['single']:.0%} | " + " | ".join(f"{r[s]:.0%}" for s in multi) + " |")
lines += ["", "Shape rules that fired on test scenes (single-blob cases): " +
          "; ".join(f"{s}: {fired[s]}" for s in SCENARIOS if fired[s])]
print("\n" + "\n".join(lines))
if args.out:
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
