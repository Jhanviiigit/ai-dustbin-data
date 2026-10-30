"""report.py <repo> <preds dir> <out.md>: metrics for every test_*.csv in the preds dir, per photo and per item,
plus paired comparisons. Labels 0 = dry, 1 = wet."""
import csv, glob, os, sys
import numpy as np
repo, pdir, out = sys.argv[1:4]
sys.path.insert(0, os.path.join(repo, "update"))
from metrics import summary, paired, wilson

def auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))

def ece(y, p_wet, bins=10):
    pred = (p_wet >= 0.5).astype(int); conf = np.where(pred == 1, p_wet, 1 - p_wet); right = pred == y
    e = 0.0
    for lo in np.linspace(0.5, 1.0, bins + 1)[:-1]:
        m = (conf >= lo) & (conf < lo + 0.5 / bins + (1e-9 if lo + 0.5 / bins >= 1 else 0))
        if m.any(): e += m.mean() * abs(right[m].mean() - conf[m].mean())
    return e

def load(f):
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    return rows[0]["model"], rows

models = dict(load(f) for f in sorted(glob.glob(os.path.join(pdir, "test_*.csv"))))
L = []
def views(rows):
    y = np.array([r["label"] == "wet" for r in rows], int); p = np.array([float(r["wet_prob"]) for r in rows])
    items = {}
    for r in rows: items.setdefault(r["item_id"], []).append(r)
    iy = np.array([v[0]["label"] == "wet" for v in items.values()], int)
    ip = np.array([np.mean([float(r["wet_prob"]) for r in v]) for v in items.values()])
    return {"photo": (y, p, [r["crop"] for r in rows]), "item": (iy, ip, list(items))}

V = {m: views(r) for m, r in models.items()}
for unit in ("photo", "item"):
    L.append(f"\n## Per {unit}\n")
    L.append("| Model | n | Errors | Accuracy [95% CI] | Balanced | Dry P / R / F1 | Wet P / R / F1 | AUC | ECE | Confusion (rows true dry, wet) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for m, v in V.items():
        y, p, _ = v[unit]; pred = (p >= 0.5).astype(int); s = summary(y, pred); lo, hi = s["accuracy_ci"]
        d, w = s["per_class"]["dry"], s["per_class"]["wet"]
        cm = [[int(((y == a) & (pred == b)).sum()) for b in (0, 1)] for a in (0, 1)]
        L.append(f"| {m} | {s['n']} | {s['errors']} | {s['accuracy']:.1%} [{lo:.1%}, {hi:.1%}] | {s['balanced_accuracy']:.1%} | "
                 f"{d['precision']:.2f} / {d['recall']:.2f} / {d['f1']:.2f} | {w['precision']:.2f} / {w['recall']:.2f} / {w['f1']:.2f} | "
                 f"{auc(y, p):.3f} | {ece(y, p):.3f} | {cm} |")
L.append("\n## Review policy (threshold 0.8, per photo)\n")
L.append("| Model | Sent to a person | Errors caught | Errors left (auto-tilted wrongly) | Accuracy of what the bin does alone |")
L.append("|---|---|---|---|---|")
for m, v in V.items():
    y, p, _ = v["photo"]; pred = (p >= 0.5).astype(int); conf = np.maximum(p, 1 - p); ask = conf < 0.8; err = pred != y
    auto = ~ask
    L.append(f"| {m} | {ask.sum()}/{len(y)} = {ask.mean():.0%} | {(err & ask).sum()}/{err.sum()} | {(err & auto).sum()} | "
             f"{(~err & auto).sum()}/{auto.sum()} = {(~err & auto).sum()/max(auto.sum(),1):.1%} |")
L.append("\n## Paired comparisons (same photos, exact McNemar)\n")
L.append("| A | B | n | only A wrong | only B wrong | both wrong | p |")
L.append("|---|---|---|---|---|---|---|")
names = list(V)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = V[names[i]]["photo"], V[names[j]]["photo"]
        common = sorted(set(a[2]) & set(b[2])); ia = {c: k for k, c in enumerate(a[2])}; ib = {c: k for k, c in enumerate(b[2])}
        y = np.array([a[0][ia[c]] for c in common]); pa = np.array([a[1][ia[c]] >= 0.5 for c in common], int); pb = np.array([b[1][ib[c]] >= 0.5 for c in common], int)
        r = paired(y, pa, pb)
        L.append(f"| {names[i]} | {names[j]} | {r['n']} | {r['only_a_wrong']} | {r['only_b_wrong']} | {r['both_wrong']} | {r['mcnemar_p']:.3f} |")
open(out, "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))
