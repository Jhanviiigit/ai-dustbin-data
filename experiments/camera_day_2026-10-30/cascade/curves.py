import csv, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os
HERE = os.path.relpath(os.path.dirname(os.path.abspath(__file__))); rep = json.load(open(os.path.join(HERE, "cascade_report.json")))
def tcurve(tf):
    r = list(csv.DictReader(open(os.path.join(HERE, "predictions", tf), encoding="utf-8"))); c = np.array([float(x["confidence"]) for x in r]); e = np.array([x["pred"] != x["label"] for x in r])
    o = np.argsort(c, kind="stable"); n = len(c); return [(k, int(e[o[k:]].sum())) for k in range(n + 1)], n
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (name, tfile, title) in zip(axes, [("pair1 baseline + teacher_v1_gray", "test_teacher_v1.csv", "Before the update: baseline + teacher_v1"),
                                            ("pair2 student_u1 + teacher_v2", "test_teacher_v2.csv", "After one update: student_u1 + teacher_v2")]):
    r = rep[name]["photo"]; n = r["n"]; t = rep[name]["tuning (loop pile)"]["t"]
    s = np.array(r["curve_student"]); b = np.array(r["curve_cascade"]); tc, _ = tcurve(tfile); tc = np.array(tc)
    ax.plot(100 * s[:, 0] / n, s[:, 1], color="#888888", label="student alone")
    ax.plot(100 * b[:, 0] / n, b[:, 1], color="#1f6fb4", lw=2, label=f"student + teacher second opinion (t = {t})")
    ax.plot(100 * tc[:, 0] / n, tc[:, 1], color="#c0622b", ls="--", label="teacher on every load")
    ax.set_xlim(0, 40); ax.set_title(title, fontsize=10); ax.set_xlabel("loads sent to a person (%)"); ax.grid(alpha=.3)
axes[0].set_ylabel("wrong decisions no person sees (of 113 photos)"); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "cascade_curves.png"), dpi=160); print("saved")
