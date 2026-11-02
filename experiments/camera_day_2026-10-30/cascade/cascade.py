"""python experiments/camera_day_2026-10-30/cascade/cascade.py

Teacher as a second opinion, session 1 only. Threshold chosen on the loop pile, result on the test pile.
Policies (one photo = one load; audits left out; the 'person' answers with the true label):
  a  student alone: confidence < 0.8 -> person
  b  cascade: student < s -> teacher; teacher >= t -> teacher's label, else person
  c  student alone, stricter, asking a person as often as b (matched)
  d  teacher on every load: confidence < 0.8 -> person"""
import csv, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "update")))
from metrics import wilson, mcnemar_exact
import os
HERE = os.path.relpath(os.path.dirname(os.path.abspath(__file__))); P = os.path.join(HERE, "predictions") + "/"
GRID = (0.80, 0.85, 0.90, 0.95, 0.99); S = 0.8
used = set(open(os.path.join(HERE, "loop_items_used.txt")).read().split())

def load(f):
    return {r["crop"]: r for r in csv.DictReader(open(P + f, encoding="utf-8"))}
def arrays(sf, tf, keep=lambda r: True):
    s, t = load(sf), load(tf); ks = sorted(k for k in s if keep(s[k]))
    g = lambda d, c: np.array([d[k][c] for k in ks])
    return dict(item=g(s, "item_id"), y=g(s, "label"), sp=g(s, "pred"), sc=g(s, "confidence").astype(float),
                tp=g(t, "pred"), tc=g(t, "confidence").astype(float))
def by_item(A):
    """one load per item: mean wet probability over its photos (confidence = max(p, 1-p))"""
    out = {k: [] for k in A}; items = sorted(set(A["item"]))
    for i in items:
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
def pick_t(A):
    unsure = A["sc"] < S
    for t in GRID:
        _, _, defer, wrong = cascade(A, t, unsure)
        if wrong.sum() == 0: return t, int(defer.sum())
    return 0.99, None
def matched(A, k):
    ask = np.zeros(len(A["y"]), bool); ask[np.argsort(A["sc"], kind="stable")[:k]] = True
    return ask, (A["sp"] != A["y"]) & ~ask

pairs = {"pair1 baseline + teacher_v1_gray": ("student_base", "teacher_v1", lambda r: True),
         "pair2 student_u1 + teacher_v2": ("student_u1", "teacher_v2", lambda r: r["item_id"] not in used)}
report = {}
for name, (s, t, keep) in pairs.items():
    tune = arrays(f"loop_{s}.csv", f"loop_{t}.csv", keep)
    tstar, ndec = pick_t(tune)
    tune_info = {"photos": len(tune["y"]), "items": len(set(tune["item"])), "unsure": int((tune["sc"] < S).sum()), "t": tstar,
                 "teacher_decided_at_t": ndec,
                 "per_t": {str(tt): [int(x.sum()) for x in cascade(tune, tt, tune["sc"] < S)[2:]] for tt in GRID}}
    res = {"tuning (loop pile)": tune_info}
    for unit in ("photo", "item"):
        A = arrays(f"test_{s}.csv", f"test_{t}.csv"); A = A if unit == "photo" else by_item(A); n = len(A["y"])
        rows = {}
        for tt in sorted({tstar, 0.95}):
            a_ask = A["sc"] < S; a_left = (A["sp"] != A["y"]) & ~a_ask
            b_ask, b_left, b_def, b_wrong = cascade(A, tt, a_ask)
            c_ask, c_left = matched(A, int(b_ask.sum()))
            d_ask = A["tc"] < S; d_left = (A["tp"] != A["y"]) & ~d_ask
            f = lambda ask, left: {"asked": f"{ask.sum()}/{n} = {ask.mean():.1%} [{wilson(int(ask.sum()), n)[0]:.1%}, {wilson(int(ask.sum()), n)[1]:.1%}]",
                                   "errors_left": int(left.sum()), "acc_alone": f"{((~left) & ~ask).sum()}/{(~ask).sum()} = {((~left) & ~ask).sum()/max((~ask).sum(),1):.1%}"}
            mc = lambda x, z: {"only_b_wrong": int((x & ~z).sum()), "only_other_wrong": int((z & ~x).sum()), "p": round(mcnemar_exact(int((x & ~z).sum()), int((z & ~x).sum())), 3)}
            rows[f"t={tt}"] = {"a student alone (0.8)": f(a_ask, a_left), "b cascade": {**f(b_ask, b_left), "teacher_decided": int(b_def.sum()), "teacher_wrong": int(b_wrong.sum())},
                               "c student alone, matched": f(c_ask, c_left), "d teacher alone (0.8)": f(d_ask, d_left),
                               "errors left, b vs a": mc(b_left, a_left), "errors left, b vs c": mc(b_left, c_left)}
        # point 2: 'unsure' = the q least-confident students loads (calibration-free), cascade at t*
        share = {}
        for q in (0.10, 0.15, 0.20, 0.25):
            k = int(round(q * n)); uns = np.zeros(n, bool); uns[np.argsort(A["sc"], kind="stable")[:k]] = True
            b_ask, b_left, b_def, b_wrong = cascade(A, tstar, uns); c_ask, c_left = matched(A, int(b_ask.sum()))
            share[f"{q:.0%}"] = {"unsure": k, "cascade asked": int(b_ask.sum()), "cascade errors left": int(b_left.sum()),
                                 "teacher decided / wrong": f"{int(b_def.sum())} / {int(b_wrong.sum())}",
                                 "student alone at same asked: errors left": int(c_left.sum()),
                                 "student alone asking all unsure: errors left": int(matched(A, k)[1].sum())}
        # point 4: curves (asked, errors left) for student alone and for the cascade, over every share
        order = np.argsort(A["sc"], kind="stable"); curve_s, curve_b = [], []
        for k in range(n + 1):
            uns = np.zeros(n, bool); uns[order[:k]] = True
            curve_s.append((k, int(((A["sp"] != A["y"]) & ~uns).sum())))
            ba, bl, _, _ = cascade(A, tstar, uns); curve_b.append((int(ba.sum()), int(bl.sum())))
        res[unit] = {"n": n, "policies": rows, "unsure by share (cascade at t*)": share,
                     "curve_student": curve_s, "curve_cascade": curve_b}
    report[name] = res
json.dump(report, open(os.path.join(HERE, "cascade_report.json"), "w"), indent=1)
for name, res in report.items():
    print("\n#####", name); print("tuning:", res["tuning (loop pile)"])
    for unit in ("photo", "item"):
        print(f"-- test pile per {unit} (n {res[unit]['n']})")
        for tt, rows in res[unit]["policies"].items():
            print("  ", tt)
            for k, v in rows.items(): print("     ", k, v)
        for q, v in res[unit]["unsure by share (cascade at t*)"].items(): print("   share", q, v)
