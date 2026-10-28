"""
metrics.py
----------
The numbers every update reports, in one place, so the gate, the teacher and the
student all compute them the same way. Labels are 0 = dry, 1 = wet. numpy only.

  summary(labels, preds)          n, errors, accuracy, balanced accuracy, per-class P/R/F1,
                                  with 95% Wilson intervals on accuracy and recall
  wilson(k, n)                    95% Wilson score interval for k successes out of n
  paired(labels, preds_a, preds_b)  on the same images: only A wrong, only B wrong, the
                                  difference in errors, and the exact McNemar p-value
"""

import math

import numpy as np

CLASS_NAMES = ("dry", "wet")
Z95 = 1.959964


def wilson(k, n, z=Z95):
    """(low, high) of the Wilson score interval; (0, 1) when n == 0."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _arrays(labels, *preds):
    out = [np.asarray(labels).astype(int).ravel()] + [np.asarray(p).astype(int).ravel() for p in preds]
    if any(len(a) != len(out[0]) for a in out):
        raise ValueError("labels and predictions differ in length")
    return out


def summary(labels, preds):
    y, p = _arrays(labels, preds)
    n = len(y)
    right = int((y == p).sum())
    out = {"n": n, "errors": n - right, "accuracy": right / n if n else float("nan"),
           "accuracy_ci": wilson(right, n), "per_class": {}}
    recalls = []
    for c, name in enumerate(CLASS_NAMES):
        tp, support, predicted = int(((y == c) & (p == c)).sum()), int((y == c).sum()), int((p == c).sum())
        prec = tp / predicted if predicted else float("nan")
        rec = tp / support if support else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if predicted and support and prec + rec else float("nan")
        out["per_class"][name] = {"support": support, "precision": prec, "recall": rec, "f1": f1,
                                  "recall_ci": wilson(tp, support)}
        if support:
            recalls.append(rec)
    out["balanced_accuracy"] = float(np.mean(recalls)) if recalls else float("nan")
    return out


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for b and c discordant pairs (binomial, p = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(labels, preds_a, preds_b):
    """A against B on the same images. extra_errors > 0 means A makes more mistakes than B."""
    y, a, b = _arrays(labels, preds_a, preds_b)
    a_right, b_right = a == y, b == y
    only_a_wrong = int((~a_right & b_right).sum())
    only_b_wrong = int((a_right & ~b_right).sum())
    return {"n": len(y), "only_a_wrong": only_a_wrong, "only_b_wrong": only_b_wrong,
            "both_wrong": int((~a_right & ~b_right).sum()), "extra_errors": only_a_wrong - only_b_wrong,
            "mcnemar_p": mcnemar_exact(only_a_wrong, only_b_wrong)}


def describe(s):
    """One line for a summary() dict."""
    lo, hi = s["accuracy_ci"]
    pc = "  ".join(f"{k} P {v['precision']:.3f} R {v['recall']:.3f} (n {v['support']})" for k, v in s["per_class"].items())
    return (f"{s['n']} images, {s['errors']} errors, acc {s['accuracy']:.4f} [{lo:.3f}, {hi:.3f}], "
            f"balanced {s['balanced_accuracy']:.4f}  |  {pc}")
