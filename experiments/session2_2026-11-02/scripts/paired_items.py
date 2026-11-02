"""Paired comparisons per item (an item's wet probability = the mean over its photos), every pair of models.
Photos of one item are not independent, so this is the stricter test; report.py gives the per-photo one.

  python paired_items.py <preds folder> >> <session>/scores_tables.md
"""
import csv
import glob
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "update")))
from metrics import paired  # noqa: E402

models = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "test_*.csv"))):
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    items = {}
    for r in rows:
        items.setdefault(r["item_id"], [int(r["label"] == "wet"), []])[1].append(float(r["wet_prob"]))
    models[rows[0]["model"]] = {k: (y, int(np.mean(p) >= 0.5)) for k, (y, p) in items.items()}

print("\n## Paired comparisons per item (exact McNemar)\n")
print("| A | B | items | only A wrong | only B wrong | both wrong | p |")
print("|---|---|---|---|---|---|---|")
for a, b in itertools.combinations(models, 2):
    keys = sorted(set(models[a]) & set(models[b]))
    r = paired([models[a][k][0] for k in keys], [models[a][k][1] for k in keys], [models[b][k][1] for k in keys])
    print(f"| {a} | {b} | {r['n']} | {r['only_a_wrong']} | {r['only_b_wrong']} | {r['both_wrong']} | {r['mcnemar_p']:.3f} |")
