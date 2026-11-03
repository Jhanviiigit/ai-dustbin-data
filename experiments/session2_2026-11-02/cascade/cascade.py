"""python experiments/session2_2026-11-02/cascade/cascade.py     (from the repo root)

Pipeline 2 (the teacher as a second opinion) on session 2, offline, from the saved predictions in
../results/predictions. The policy functions are the ones in experiments/camera_day_2026-10-30/cascade/cascade.py
(branch results/teacher-second-opinion), unchanged. The teacher threshold t is FIXED from session 1
(pair 1: 0.90, pair 2: 0.99; 0.95 is reported as well); nothing is chosen or tuned on session 2.

Policies (one photo = one load; audits left out; the 'person' answers with the true label):
  pipeline 1 (a)  student alone: confidence < 0.8 -> person
  pipeline 2 (b)  student < 0.8 -> teacher; teacher confidence >= t -> the teacher's label, else person
  c               student alone, made stricter, asking a person as often as pipeline 2 (matched)
  d               teacher on every load: confidence < 0.8 -> person
"""
import csv, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "update")))
from metrics import wilson, mcnemar_exact
P = os.path.join(HERE, "..", "results", "predictions") + os.sep
S = 0.8

def load(f): return {r["crop"]: r for r in csv.DictReader(open(P + f, encoding="utf-8"))}
def arrays(sf, tf):
    s, t = load(sf), load(tf); ks = sorted(k for k in s if k in t)
    g = lambda d, c: np.array([d[k][c] for k in ks])
    return dict(item=g(s, "item_id"), y=g(s, "label"), sp=g(s, "pred"), sc=g(s, "confidence").astype(float),
                tp=g(t, "pred"), tc=g(t, "confidence").astype(float))
def by_item(A):
    out = {k: [] for k in A}
    for i in sorted(set(A["item"])):
        m = A["item"] == i
        def mean_wet(pred, conf): return np.mean(np.where(pred[m] == "wet", conf[m], 1 - conf[m]))
        ps, pt = mean_wet(A["sp"], A["sc"]), mean_wet(A["tp"], A["tc"])
        out["item"].append(i); out["y"].append(A["y"][m][0])
        out["sp"].append("wet" if ps >= .5 else "dry"); out["sc"].append(max(ps, 1 - ps))
        out["tp"].append("wet" if pt >= .5 else "dry"); out["tc"].append(max(pt, 1 - pt))
    return {k: np.array(v) for k, v in out.items()}
def cascade(A, t, unsure):
    defer = unsure & (A["tc"] >= t); ask = unsure & ~defer
    final = np.where(defer, A["tp"], A["sp"]); left = (final != A["y"]) & ~ask
    return ask, left, defer, (A["tp"] != A["y"]) & defer
def matched(A, k):
    ask = np.zeros(len(A["y"]), bool); ask[np.argsort(A["sc"], kind="stable")[:k]] = True
    return ask, (A["sp"] != A["y"]) & ~ask

pairs = {"pair1 baseline + teacher_v1_gray": ("student_base", "teacher_v1", 0.90, "Before the update: baseline + teacher_v1"),
         "pair2 student_u1 + teacher_v2": ("student_u1", "teacher_v2", 0.99, "After one update: student_u1 + teacher_v2")}
report = {}
for name, (s, t, tstar, _) in pairs.items():
    res = {"t_fixed_from_session1": tstar}
    for unit in ("photo", "item"):
        A = arrays(f"test_{s}.csv", f"test_{t}.csv"); A = A if unit == "photo" else by_item(A); n = len(A["y"])
        rows = {}
        for tt in sorted({tstar, 0.95}):
            a_ask = A["sc"] < S; a_left = (A["sp"] != A["y"]) & ~a_ask
            b_ask, b_left, b_def, b_wrong = cascade(A, tt, a_ask)
            c_ask, c_left = matched(A, int(b_ask.sum()))
            d_ask = A["tc"] < S; d_left = (A["tp"] != A["y"]) & ~d_ask
            f = lambda ask, left: {"asked": f"{ask.sum()}/{n} = {ask.mean():.1%} [{wilson(int(ask.sum()), n)[0]:.1%}, {wilson(int(ask.sum()), n)[1]:.1%}]", "errors_left": int(left.sum())}
            mc = lambda x, z: {"only_b_wrong": int((x & ~z).sum()), "only_other_wrong": int((z & ~x).sum()), "p": round(mcnemar_exact(int((x & ~z).sum()), int((z & ~x).sum())), 3)}
            rows[f"t={tt}"] = {"a student alone (0.8)": f(a_ask, a_left),
                               "b cascade": {**f(b_ask, b_left), "teacher_decided": int(b_def.sum()), "teacher_wrong": int(b_wrong.sum())},
                               "c student alone, matched": f(c_ask, c_left), "d teacher alone (0.8)": f(d_ask, d_left),
                               "errors left, b vs a": mc(b_left, a_left), "errors left, b vs c": mc(b_left, c_left)}
        share = {}                                   # 'unsure' = the q least-confident student loads (no calibration involved)
        for q in (0.10, 0.15, 0.20, 0.25):
            k = int(round(q * n)); uns = np.zeros(n, bool); uns[np.argsort(A["sc"], kind="stable")[:k]] = True
            ba, bl, bd, bw = cascade(A, tstar, uns); ca, cl = matched(A, int(ba.sum()))
            share[f"{q:.0%}"] = {"unsure": k, "pipeline2 asked": int(ba.sum()), "pipeline2 errors left": int(bl.sum()),
                                 "teacher decided / wrong": f"{int(bd.sum())} / {int(bw.sum())}",
                                 "pipeline1 at the same asked: errors left": int(cl.sum())}
        order = np.argsort(A["sc"], kind="stable"); cs, cb = [], []
        for k in range(n + 1):
            uns = np.zeros(n, bool); uns[order[:k]] = True
            cs.append((k, int(((A["sp"] != A["y"]) & ~uns).sum()))); ba, bl, _, _ = cascade(A, tstar, uns); cb.append((int(ba.sum()), int(bl.sum())))
        res[unit] = {"n": n, "policies": rows, "unsure by share (cascade at t fixed)": share, "curve_pipeline1": cs, "curve_pipeline2": cb}
    report[name] = res
json.dump(report, open(os.path.join(HERE, "cascade_report.json"), "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (name, (s, t, tstar, title)) in zip(axes, pairs.items()):
    r = report[name]["photo"]; n = r["n"]; c1, c2 = np.array(r["curve_pipeline1"]), np.array(r["curve_pipeline2"])
    T = arrays(f"test_{t}.csv", f"test_{t}.csv"); o = np.argsort(T["sc"], kind="stable"); e = T["sp"] != T["y"]
    tc = np.array([(k, int(e[o[k:]].sum())) for k in range(n + 1)])
    ax.plot(100 * c1[:, 0] / n, c1[:, 1], color="#888888", label="pipeline 1: student alone")
    ax.plot(100 * c2[:, 0] / n, c2[:, 1], color="#1f6fb4", lw=2, label=f"pipeline 2: + teacher second opinion (t = {tstar})")
    ax.plot(100 * tc[:, 0] / n, tc[:, 1], color="#c0622b", ls="--", label="teacher on every load")
    ax.set_xlim(0, 40); ax.set_title(title, fontsize=10); ax.set_xlabel("loads sent to a person (%)"); ax.grid(alpha=.3)
axes[0].set_ylabel("wrong decisions no person sees (of 148 photos)"); axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "cascade_curves.png"), dpi=150)
for name, res in report.items():
    print("\n#####", name, "t fixed", res["t_fixed_from_session1"])
    for unit in ("photo", "item"):
        print(f"-- per {unit} (n {res[unit]['n']})")
        for tt, rows in res[unit]["policies"].items():
            print("  ", tt)
            for k, v in rows.items(): print("     ", k, v)
        for q, v in res[unit]["unsure by share (cascade at t fixed)"].items(): print("   share", q, v)
