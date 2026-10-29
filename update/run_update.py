"""
run_update.py
-------------
One full update, start to finish. Run it again at any point: it carries on where it
stopped (the state is kept in pool/updates/update_N.json under "run").

  1. ingest    new feedback -> pool (ingest.py), if the trigger says go
  2. teacher   wait for the teacher's soft labels for this update (run on Colab; see below)
               and check they cover every train image
  3. student   update_student.py: the current student + pool/train + soft labels -> candidate
  4. gate      promote.py on test_fixed / test_audit (test_hard reported)
  5. deploy    accepted: deploy/current/{weights.pt, info.json}; the previous one -> deploy/previous
  6. record    one row in models.csv, accepted or rejected

  python update/run_update.py --pool pool --feedback feedback --current deploy/current/weights.pt
  python update/run_update.py --pool pool                       # carry on (e.g. after the teacher ran)

Exit codes: 0 finished (deployed or rejected), 2 waiting for more feedback, 3 waiting for the
teacher's soft labels, 1 error.

Names: update N trains student_uN with teacher_v(N+1)'s soft labels (teacher_v1 is the public-data
teacher), expected at
<runs>/teacher_v(N+1)/soft_labels.csv (columns crop_id, teacher_logit, teacher_wet_prob, fold,
teacher_version). Rejected candidates keep their name, so a name is never reused.

The threshold the bin uses comes from the student's update.json ("threshold", picked on val);
until the student writes one, THRESHOLD_FALLBACK (0.8) is used and models.csv says so.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from ingest import ingest  # noqa: E402
from ledger import Ledger  # noqa: E402
from pool import Pool  # noqa: E402

THRESHOLD_FALLBACK = 0.8     # same as bin_vision/hitl.py
TEACHER_OFFSET = 1          # teacher_v1 is the public-data teacher, so update N uses teacher_v(N+1)
WAIT_FEEDBACK, WAIT_TEACHER = 2, 3
STAGES = ("ingested", "teacher_ready", "student_trained", "gated", "finished")
MODELS_COLUMNS = ["update", "date", "version", "status", "replaces", "teacher_version", "pool_train_images",
                  "pool_train_groups", "new_rows", "test_fixed_n", "test_fixed_errors", "test_fixed_acc",
                  "test_fixed_balanced_acc", "best_shipped", "extra_errors_vs_best", "test_audit_n",
                  "test_audit_acc", "test_audit_gated", "test_hard_n", "test_hard_acc", "temperature",
                  "threshold", "threshold_source", "reasons"]


class Update:
    """pool/updates/update_N.json, with its "run" section."""

    def __init__(self, pool_dir, number):
        self.number, self.name = number, f"update_{number}"
        self.path = os.path.join(pool_dir, "updates", f"{self.name}.json")
        with open(self.path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.run = self.data.setdefault("run", {"stage": "ingested"})

    def done(self, stage, **info):
        self.run.update(info, stage=stage)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, self.path)

    def at_least(self, stage):
        return STAGES.index(self.run["stage"]) >= STAGES.index(stage)


def call(script, *args, log=None):
    cmd = [sys.executable, script, *map(str, args)]
    print("\n$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace")
    print(proc.stdout, end="")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write("$ " + " ".join(cmd) + "\n" + proc.stdout)
    return proc.returncode


def check_soft_labels(path, pool):
    """Problems with a soft-label file for this pool (empty list = fine)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    need = {"crop_id", "teacher_wet_prob", "teacher_version"}
    if not rows or not need <= set(rows[0]):
        return [f"{path}: needs columns {sorted(need)}"]
    have = {r["crop_id"] for r in rows}
    train = {r["crop_id"] for r in pool.rows_in("train")}
    problems = []
    if train - have:
        problems.append(f"{len(train - have)} train images have no soft label, e.g. {sorted(train - have)[:3]}")
    held_out = have & {r["crop_id"] for r in pool.rows if r["split"] != "train"}
    if held_out:
        problems.append(f"{len(held_out)} soft labels are for val/test images, which must not be trained on")
    bad = [r["crop_id"] for r in rows if not 0.0 <= float(r["teacher_wet_prob"]) <= 1.0]
    if bad:
        problems.append(f"{len(bad)} teacher_wet_prob values are not probabilities")
    return problems


def latest(pool_dir):
    n = Ledger(pool_dir).updates
    return Update(pool_dir, n) if n and os.path.exists(os.path.join(pool_dir, "updates", f"update_{n}.json")) else None


def models_row(update, teacher_version, new_rows, gate, status, threshold, source, pool):
    sets = gate.get("sets", {})

    def get(name, *keys):
        x = sets.get(name, {})
        for k in keys:
            x = x.get(k, {}) if isinstance(x, dict) else {}
        return "" if x in ({}, None) else x

    def acc(name, key="accuracy"):
        v = get(name, "candidate", key)
        return f"{v:.4f}" if isinstance(v, float) else ""

    train = pool.rows_in("train")
    return {"update": update, "date": datetime.now().isoformat(timespec="seconds"), "version": gate.get("version", ""),
            "status": status, "replaces": gate.get("current") or "", "teacher_version": teacher_version or "",
            "pool_train_images": len(train), "pool_train_groups": len({r["group_id"] for r in train}),
            "new_rows": new_rows, "test_fixed_n": get("test_fixed", "n"),
            "test_fixed_errors": get("test_fixed", "candidate", "errors"), "test_fixed_acc": acc("test_fixed"),
            "test_fixed_balanced_acc": acc("test_fixed", "balanced_accuracy"), "best_shipped": get("test_fixed", "best"),
            "extra_errors_vs_best": get("test_fixed", "vs_best", "extra_errors"), "test_audit_n": get("test_audit", "n"),
            "test_audit_acc": acc("test_audit"), "test_audit_gated": get("test_audit", "gated"),
            "test_hard_n": get("test_hard", "n"), "test_hard_acc": acc("test_hard"),
            "temperature": gate.get("temperature", ""), "threshold": threshold, "threshold_source": source,
            "reasons": "; ".join(gate.get("reasons", []))}


def append_models_csv(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MODELS_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def deploy(weights, deploy_dir, info):
    """deploy/current <- the new weights; the old current -> deploy/previous (one-step rollback)."""
    cur, prev = os.path.join(deploy_dir, "current"), os.path.join(deploy_dir, "previous")
    try:
        with open(os.path.join(cur, "info.json"), encoding="utf-8") as f:
            if json.load(f).get("version") == info["version"]:
                return os.path.join(cur, "weights.pt")         # already deployed (a re-run after a crash)
    except (OSError, ValueError):
        pass
    staging = os.path.join(deploy_dir, "incoming")
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    shutil.copy2(weights, os.path.join(staging, "weights.pt"))
    with open(os.path.join(staging, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    if os.path.isdir(cur):
        shutil.rmtree(prev, ignore_errors=True)
        os.replace(cur, prev)
    os.replace(staging, cur)
    return os.path.join(cur, "weights.pt")


def run(a):
    pool_dir = os.path.abspath(a.pool)
    u = latest(pool_dir)

    # 1. ingest (only when the last update has finished)
    if u is None or u.at_least("finished"):
        if not a.feedback:
            print("nothing in progress, and no --feedback to ingest")
            return WAIT_FEEDBACK
        report, _ = ingest(pool_dir, a.feedback, commit=True, force=a.force)
        print(json.dumps({k: report[k] for k in ("new_rows", "plan", "trigger")}, indent=1))
        if not report["committed"]:
            print(f"waiting for feedback: {report['trigger']['reason']}")
            return WAIT_FEEDBACK
        u = latest(pool_dir)
        current = os.path.abspath(a.current or os.path.join(a.deploy, "current", "weights.pt"))
        if not os.path.isfile(current):
            raise SystemExit(f"no current student at {current} (give --current the first time)")
        u.done("ingested", current=current, started=datetime.now().isoformat(timespec="seconds"))
    else:
        print(f"carrying on with {u.name} (stage: {u.run['stage']})")
    pool = Pool(pool_dir)
    runs = os.path.abspath(a.runs)
    teacher_version, version = f"teacher_v{u.number + TEACHER_OFFSET}", f"student_u{u.number}"
    out = os.path.join(runs, version)
    os.makedirs(out, exist_ok=True)
    log = os.path.join(out, "run_update.log")

    # 2. the teacher's soft labels
    if not u.at_least("teacher_ready"):
        soft = os.path.abspath(a.soft_labels or os.path.join(runs, teacher_version, "soft_labels.csv"))
        if not os.path.isfile(soft):
            print(f"\nwaiting for the teacher: {u.name} is in the pool. Run the teacher on Colab with\n"
                  f"  version {teacher_version}, writing {soft}\n"
                  f"then run this again (same --pool) to carry on.")
            return WAIT_TEACHER
        problems = check_soft_labels(soft, pool)
        if problems:
            raise SystemExit("soft labels not usable:\n  " + "\n  ".join(problems))
        u.done("teacher_ready", soft_labels=soft, teacher_version=teacher_version)

    # 3. the student
    if not u.at_least("student_trained"):
        code = call(a.student_script, "--current", u.run["current"], "--version", version, "--pool", pool_dir,
                    "--soft_labels", u.run["soft_labels"], "--out", out, "--force", log=log)
        if code:
            raise SystemExit(f"update_student.py failed (exit {code}); see {log}")
        u.done("student_trained", candidate=os.path.join(out, "weights.pt"))

    # 4. the gate
    gate_json = os.path.join(out, "gate.json")
    if not u.at_least("gated"):
        code = call(os.path.join(REPO, "student", "promote.py"), "--candidate", u.run["candidate"], "--current",
                    u.run["current"], "--pool", pool_dir, "--releases", a.releases, "--report", gate_json, log=log)
        if not os.path.exists(gate_json):
            raise SystemExit(f"promote.py wrote no report (exit {code}); see {log}")
        with open(gate_json, encoding="utf-8") as f:
            accepted = json.load(f)["accepted"]
        u.done("gated", accepted=accepted)

    # 5-6. deploy and record
    with open(gate_json, encoding="utf-8") as f:
        gate = json.load(f)
    threshold, source = THRESHOLD_FALLBACK, f"fallback {THRESHOLD_FALLBACK}"
    upd_json = os.path.join(out, "update.json")
    if os.path.exists(upd_json):
        with open(upd_json, encoding="utf-8") as f:
            t = json.load(f).get("threshold")
        if t is not None:
            threshold, source = float(t), "val (update.json)"
    status = "rejected"
    if gate["accepted"]:
        released = os.path.join(a.releases, version, "weights.pt")
        info = {"version": version, "update": u.name, "teacher_version": u.run.get("teacher_version"),
                "threshold": threshold, "threshold_source": source, "temperature": gate.get("temperature"),
                "deployed": datetime.now().isoformat(timespec="seconds"), "replaces": gate.get("current")}
        path = deploy(released, a.deploy, info)
        status = "deployed"
        print(f"\nDEPLOYED {version} -> {path} (threshold {threshold}, {source})")
    else:
        print(f"\nREJECTED {version}: " + "; ".join(gate["reasons"]) + "\nthe bin keeps its current student")
    append_models_csv(a.models_csv, models_row(u.name, u.run.get("teacher_version"), u.data.get("new_rows", ""), gate,
                                               status, threshold, source, pool))
    u.done("finished", status=status, finished=datetime.now().isoformat(timespec="seconds"))
    print(f"recorded in {a.models_csv}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="One full update: feedback -> pool -> teacher -> student -> gate -> deploy")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--feedback", nargs="*", default=[], help="hitl.py feedback folder(s)")
    ap.add_argument("--current", help="the student on the bin now (default: <deploy>/current/weights.pt)")
    ap.add_argument("--force", action="store_true", help="ingest even if the trigger says wait")
    ap.add_argument("--soft_labels", help="the teacher's soft labels (default: <runs>/teacher_v(N+1)/soft_labels.csv)")
    ap.add_argument("--runs", default="runs", help="where teacher_vN/ and student_uN/ live")
    ap.add_argument("--releases", default=os.path.join(REPO, "student", "releases"))
    ap.add_argument("--deploy", default="deploy")
    ap.add_argument("--models_csv", default="models.csv")
    ap.add_argument("--student_script", default=os.path.join(REPO, "student", "update_student.py"))
    a = ap.parse_args()
    a.releases, a.deploy, a.models_csv = map(os.path.abspath, (a.releases, a.deploy, a.models_csv))
    sys.exit(run(a))
