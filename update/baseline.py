"""
baseline.py
-----------
Ship the first student, once, before any update: the one every later student is compared with.

  python update/baseline.py --pool pool --weights student/runs/organik/weights.pt

  1. gate      promote.py as a first release (nothing shipped yet, so only the calibration
               check applies); it scores test_fixed / test_audit / test_hard for the record
  2. release   <releases>/<version>/weights.pt + info.json
  3. deploy    <deploy>/current/{weights.pt, info.json}, with the threshold (--threshold,
               default THRESHOLD_FALLBACK 0.8) that the bin reads (bin_vision/hitl.py load_deployed)
  4. record    the first models.csv row (update "seed")

Refuses to run if anything was shipped or deployed already: after the baseline, new students
come only through run_update.py and its gate.

The baseline is organik_gray_v2 (release student-organik_gray_v2): trained on the Organik train
split only, which is pool/train; it did not train on pool/val or pool/test_fixed. (pool/val picked its
best epoch, epoch 6; pool/test_fixed was not used for any choice of the model.)
"""

import argparse
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from pool import Pool  # noqa: E402
from run_update import THRESHOLD_FALLBACK, append_models_csv, call, deploy, models_row  # noqa: E402


def main(a):
    if os.path.isdir(a.releases) and any(os.path.isfile(os.path.join(a.releases, d, "weights.pt"))
                                         for d in os.listdir(a.releases)):
        raise SystemExit(f"{a.releases} already holds a shipped student: the baseline is set once")
    if os.path.exists(os.path.join(a.deploy, "current")):
        raise SystemExit(f"{a.deploy}/current exists: the baseline is set once")
    pool = Pool(a.pool)
    if not pool.rows_in("test_fixed"):
        raise SystemExit(f"{a.pool} has no test_fixed images: seed it first (pool.py seed)")

    os.makedirs(a.releases, exist_ok=True)
    report = os.path.join(a.releases, "baseline_gate.json")
    code = call(os.path.join(REPO, "student", "promote.py"), "--candidate", a.weights, "--pool", a.pool,
                "--releases", a.releases, "--report", report)
    with open(report, encoding="utf-8") as f:
        gate = json.load(f)
    if code or not gate["accepted"]:
        raise SystemExit("baseline refused by the gate: " + "; ".join(gate.get("reasons", [])))

    version = gate["version"]
    source = "--threshold" if a.threshold is not None else f"fallback {THRESHOLD_FALLBACK}"
    threshold = a.threshold if a.threshold is not None else THRESHOLD_FALLBACK
    info = {"version": version, "update": "seed", "teacher_version": None, "threshold": threshold,
            "threshold_source": source, "temperature": gate.get("temperature"),
            "deployed": datetime.now().isoformat(timespec="seconds"), "replaces": None}
    path = deploy(os.path.join(a.releases, version, "weights.pt"), a.deploy, info)
    append_models_csv(a.models_csv, models_row("seed", "", len(pool.rows), gate, "deployed", threshold, source, pool))
    print(f"\nBASELINE {version} -> {path} (threshold {threshold}, {source}); recorded in {a.models_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ship the first student (the baseline)")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--weights", required=True, help="the calibrated first student")
    ap.add_argument("--threshold", type=float, default=None, help=f"review threshold (default {THRESHOLD_FALLBACK})")
    ap.add_argument("--releases", default=os.path.join(REPO, "student", "releases"))
    ap.add_argument("--deploy", default="deploy")
    ap.add_argument("--models_csv", default="models.csv")
    a = ap.parse_args()
    a.releases, a.deploy, a.models_csv = map(os.path.abspath, (a.releases, a.deploy, a.models_csv))
    main(a)
