"""
test_run_update.py
------------------
Tests for phase 3: metrics.py, the release gate (student/promote.py) and run_update.py,
on generated images. The "models" are MobileNetV2s whose head always answers one class,
so their errors are known exactly; the student step is a stand-in script with the agreed
update_student.py arguments. No datasets, no downloads.

  python update/test_run_update.py
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "student"))
from common import build_model  # noqa: E402
from metrics import paired, summary, wilson  # noqa: E402
from pool import Pool  # noqa: E402
from PIL import Image  # noqa: E402

failures = []
TMP = tempfile.mkdtemp(prefix="run_update_test_")
PROMOTE = os.path.join(REPO, "student", "promote.py")
RUN = os.path.join(HERE, "run_update.py")


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + what)
    if not cond:
        failures.append(what)


def image(seed):
    """A random blocky picture (as in test_ingest.py)."""
    r = np.random.default_rng(seed)
    return Image.fromarray(r.integers(0, 255, (8, 8, 3), dtype=np.uint8)).resize((96, 96), Image.NEAREST)


def make_feedback(name, loads):
    """loads: [(load_id, trigger, [(label, image, box, from_drop), ...])] -> a hitl.py feedback folder."""
    d = os.path.join(TMP, name)
    os.makedirs(os.path.join(d, "crops"), exist_ok=True)
    rows = []
    for load_id, trigger, items in loads:
        for i, (label, img, box, from_drop) in enumerate(items, start=1):
            cid = f"{load_id}-{i}"
            img.save(os.path.join(d, "crops", f"{cid}.jpg"), quality=95)
            rows.append({"crop_id": cid, "load_id": load_id, "timestamp": "", "student_version": "s1", "item_index": i,
                         "box": box, "student_label": label, "student_confidence": "0.55", "load_composition": "",
                         "decision": "", "trigger": trigger, "human_label": label, "crop_file": f"crops/{cid}.jpg",
                         "from_drop": from_drop})
    with open(os.path.join(d, "feedback.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return d


def constant_model(path, version, answer, calibrated=True):
    """Weights whose head always answers `answer` ('dry' or 'wet')."""
    torch.manual_seed(0)
    m = build_model(pretrained=False)
    with torch.no_grad():
        m.classifier[1].weight.zero_()
        m.classifier[1].bias.copy_(torch.tensor([50.0, 0.0] if answer == "dry" else [0.0, 50.0]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": m.state_dict(), "version": version, "variant": "bin", "calibrated": calibrated,
                "temperature": 1.0, "calibrated_on": "pool/val", "img_size": 64}, path)
    return path


def make_pool(root, n_test_dry=8, n_test_wet=2, n_audit=0):
    pool = Pool(root)
    k = 0
    for split, labels in (("train", ["dry", "wet"] * 3), ("val", ["dry", "wet"]),
                          ("test_fixed", ["dry"] * n_test_dry + ["wet"] * n_test_wet)):
        for lab in labels:
            k += 1
            pool.add(image(1000 + k), f"cap{k}", split, lab, f"item{k}", "capture", added_in="seed")
    for i in range(n_audit):
        pool.add(image(6000 + i), f"aud{i}", "test_audit", "wet", f"A{i}", "feedback", trigger="audit", from_drop="1")
    pool.save()
    return pool


def promote(*args):
    p = subprocess.run([sys.executable, PROMOTE, *map(str, args)], capture_output=True, text=True, encoding="utf-8")
    return p.returncode, p.stdout + p.stderr


def report(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── 1. metrics.py ────────────────────────────────────────────────────────────
print("Metrics:")
lo, hi = wilson(28, 30)
check(abs(lo - 0.7868) < 1e-3 and abs(hi - 0.9815) < 1e-3, f"Wilson 28/30 = [{lo:.4f}, {hi:.4f}] (textbook [0.787, 0.982])")
check(wilson(0, 0) == (0.0, 1.0), "Wilson with no data is [0, 1]")
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
s = summary(y, np.zeros(10))
check(s["errors"] == 2 and s["accuracy"] == 0.8 and s["balanced_accuracy"] == 0.5,
      "always-dry on 8 dry + 2 wet: 2 errors, acc 0.8, balanced acc 0.5")
check(s["per_class"]["dry"]["precision"] == 0.8 and s["per_class"]["wet"]["recall"] == 0.0, "per-class precision/recall")
pr = paired(y, np.ones(10), np.zeros(10))
check(pr["only_a_wrong"] == 8 and pr["only_b_wrong"] == 2 and pr["extra_errors"] == 6, "paired counts and extra errors")
check(paired(y, y, y)["mcnemar_p"] == 1.0 and paired(y, np.ones(10), y)["mcnemar_p"] < 0.01, "McNemar p-value")

# ─── 2. The gate (promote.py) ─────────────────────────────────────────────────
print("Gate:")
pool = make_pool(os.path.join(TMP, "gpool"))
M = os.path.join(TMP, "models")
dry1 = constant_model(f"{M}/dry1.pt", "s_dry1", "dry")           # 2 errors on test_fixed
dry2 = constant_model(f"{M}/dry2.pt", "s_dry2", "dry")
wet = constant_model(f"{M}/wet.pt", "s_wet", "wet")              # 8 errors
uncal = constant_model(f"{M}/uncal.pt", "s_uncal", "dry", calibrated=False)
rel = os.path.join(TMP, "releases")

code, out = promote("--candidate", dry1, "--pool", pool.root, "--releases", rel, "--dry_run")
check(code == 0 and "first release" in out, "first release: accepted on the calibration check alone")
code, out = promote("--candidate", uncal, "--pool", pool.root, "--releases", rel, "--dry_run")
check(code == 1 and "not calibrated" in out, "uncalibrated candidate rejected")
code, out = promote("--candidate", dry2, "--current", dry1, "--pool", pool.root, "--releases", rel,
                    "--report", f"{TMP}/r1.json")
r = report(f"{TMP}/r1.json")
check(code == 0 and r["sets"]["test_fixed"]["vs_current"]["extra_errors"] == 0, "same errors as current: accepted")
check(os.path.exists(f"{rel}/s_dry2/weights.pt") and os.path.exists(f"{rel}/s_dry2/info.json"), "accepted weights copied to releases/")
code, out = promote("--candidate", wet, "--current", dry2, "--pool", pool.root, "--releases", rel,
                    "--report", f"{TMP}/r2.json")
r = report(f"{TMP}/r2.json")
check(code == 1 and r["sets"]["test_fixed"]["vs_best"]["extra_errors"] == 6 and not os.path.exists(f"{rel}/s_wet"),
      "6 more errors: rejected, nothing copied")
code, out = promote("--candidate", dry1, "--current", dry2, "--pool", pool.root, "--releases", rel, "--dry_run")
check(code == 0, "a new name with the same errors passes")
code, out = promote("--candidate", dry2, "--current", dry1, "--pool", pool.root, "--releases", rel, "--dry_run")
check(code == 1 and "already shipped" in out, "a shipped version name is refused")

# drift: the current is bad already; the candidate matches it but is far behind the best shipped
wet2 = constant_model(f"{M}/wet2.pt", "s_wet2", "wet")
code, out = promote("--candidate", wet2, "--current", wet, "--pool", pool.root, "--releases", rel,
                    "--report", f"{TMP}/r3.json")
r = report(f"{TMP}/r3.json")["sets"]["test_fixed"]
check(code == 1 and r["vs_current"]["extra_errors"] == 0 and r["best"] == "s_dry2" and r["vs_best"]["extra_errors"] == 6,
      "no worse than current but 6 worse than the best shipped: rejected (no slow drift)")

# a 3-error margin: 8 dry + 5 wet in test_fixed -> always-dry 5 errors, always-wet 8 errors
p3 = make_pool(os.path.join(TMP, "p3"), 8, 5)
code, _ = promote("--candidate", wet2, "--current", dry1, "--pool", p3.root, "--releases", os.path.join(TMP, "none"), "--dry_run")
check(code == 1, "3 more errors: rejected")
p2 = make_pool(os.path.join(TMP, "p2"), 7, 5)
code, _ = promote("--candidate", wet2, "--current", dry1, "--pool", p2.root, "--releases", os.path.join(TMP, "none"), "--dry_run")
check(code == 0, "2 more errors: accepted (the allowed margin)")

# test_audit gates only from 30 images; test_hard never
pa = make_pool(os.path.join(TMP, "pa"), 5, 5, n_audit=29)
code, out = promote("--candidate", wet2, "--current", dry1, "--pool", pa.root, "--releases", os.path.join(TMP, "none"),
                    "--report", f"{TMP}/ra.json", "--dry_run")
check(code == 0 and report(f"{TMP}/ra.json")["sets"]["test_audit"]["gated"] is False,
      "29 audits: test_audit reported, not gated")
pb = make_pool(os.path.join(TMP, "pb"), 5, 5, n_audit=30)
code, out = promote("--candidate", dry2, "--current", wet, "--pool", pb.root, "--releases", os.path.join(TMP, "none"),
                    "--report", f"{TMP}/rb.json", "--dry_run")
rb = report(f"{TMP}/rb.json")
check(code == 1 and rb["sets"]["test_audit"]["gated"] and "test_audit" in " ".join(rb["reasons"]),
      "30 audits: test_audit gated (30 more errors on audits -> rejected)")
ph = Pool(os.path.join(TMP, "ph"))
make_pool(ph.root, 5, 5)
ph = Pool(ph.root)
for i in range(10):
    ph.add(image(7000 + i), f"hard{i}", "test_hard", "wet", f"H{i}", "feedback", trigger="low_confidence", from_drop="1")
ph.save()
code, _ = promote("--candidate", dry2, "--current", wet, "--pool", ph.root, "--releases", os.path.join(TMP, "none"),
                  "--report", f"{TMP}/rh.json", "--dry_run")
check(code == 0 and report(f"{TMP}/rh.json")["sets"]["test_hard"]["gated"] is False, "test_hard is reported, never gated")
dd = os.path.join(TMP, "dataset_bin")
shutil.copytree(os.path.join(p3.root, "test_fixed"), os.path.join(dd, "test"))
code, out = promote("--candidate", wet2, "--current", dry1, "--data_dir", dd, "--dry_run")
check(code == 1 and "REJECTED" in out and "3 more errors" in out, "--data_dir mode (dataset_bin/test, no pool) still works")

# ─── 3. run_update.py ─────────────────────────────────────────────────────────
print("run_update:")
W = os.path.join(TMP, "work")
os.makedirs(W)
FAKE = os.path.join(TMP, "fake_update_student.py")
with open(FAKE, "w", encoding="utf-8") as f:
    f.write('''import argparse, json, os, torch
p = argparse.ArgumentParser()
for a in ("--current", "--version", "--pool", "--soft_labels", "--out"):
    p.add_argument(a, required=True)
p.add_argument("--force", action="store_true")
a = p.parse_args()
ck = torch.load(a.current, weights_only=True)
answer = os.environ["FAKE_ANSWER"]
b = ck["state_dict"]["classifier.1.bias"]
b.copy_(torch.tensor([50.0, 0.0] if answer == "dry" else [0.0, 50.0]))
ck.update(version=a.version, calibrated=True, teacher_version="teacher_u")
os.makedirs(a.out, exist_ok=True)
torch.save(ck, os.path.join(a.out, "weights.pt"))
if os.environ.get("FAKE_THRESHOLD"):
    json.dump({"threshold": float(os.environ["FAKE_THRESHOLD"])}, open(os.path.join(a.out, "update.json"), "w"))
''')
rpool = make_pool(os.path.join(W, "pool"))
first = constant_model(os.path.join(W, "student_v1.pt"), "student_v1", "dry")
loads = [(f"L{i:03d}", "low_confidence", [("dry" if i % 3 else "wet", image(i), "10 10 40 40", 1)]) for i in range(60)]
fb = make_feedback("fb", loads)
base = ["--pool", rpool.root, "--runs", f"{W}/runs", "--releases", f"{W}/releases", "--deploy", f"{W}/deploy",
        "--models_csv", f"{W}/models.csv", "--student_script", FAKE]


def run_update(*extra, answer="dry", threshold=None):
    env = {**os.environ, "FAKE_ANSWER": answer}
    if threshold:
        env["FAKE_THRESHOLD"] = str(threshold)
    p = subprocess.run([sys.executable, RUN, *base, *extra], capture_output=True, text=True, encoding="utf-8", env=env)
    return p.returncode, p.stdout + p.stderr


def write_soft(update_n, skip=0, extra_ids=()):
    pool = Pool(rpool.root)
    d = os.path.join(W, "runs", f"teacher_u{update_n}")
    os.makedirs(d, exist_ok=True)
    rows = [r["crop_id"] for r in pool.rows_in("train")][skip:] + list(extra_ids)
    with open(os.path.join(d, "soft_labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["crop_id", "teacher_logit", "teacher_wet_prob", "fold", "teacher_version"])
        for c in rows:
            w.writerow([c, 0.0, 0.5, 0, f"teacher_u{update_n}"])


code, out = run_update("--feedback", fb, "--current", first)
check(code == 3 and "waiting for the teacher" in out, "update 1 ingested, then waits for the teacher (exit 3)")
check(os.path.exists(os.path.join(rpool.root, "updates", "update_1.json")), "update_1 is in the pool")
write_soft(1, skip=2)
code, out = run_update()
check(code == 1 and "no soft label" in out, "soft labels missing train images: refused")
write_soft(1, extra_ids=["cap7"])                                  # cap7 is a val image
code, out = run_update()
check(code == 1 and "val/test images" in out, "soft labels for val/test images: refused")
write_soft(1)
code, out = run_update(threshold=0.83)
check(code == 0 and "DEPLOYED student_u1" in out, "carries on after the teacher: student, gate, deploy")
with open(os.path.join(W, "deploy", "current", "info.json"), encoding="utf-8") as f:
    info = json.load(f)
check(info["version"] == "student_u1" and info["threshold"] == 0.83, "deploy/current has the new student and its threshold")
check(os.path.exists(os.path.join(W, "releases", "student_u1", "weights.pt")), "released into releases/")
code, out = run_update()
check(code == 2, "nothing new: waits for feedback (exit 2), deploys nothing")

fb2 = make_feedback("fb2", [(f"M{i:03d}", "low_confidence", [("dry" if i % 3 else "wet", image(2000 + i), "10 10 40 40", 1)])
                            for i in range(60)])
code, out = run_update("--feedback", fb2, answer="wet")
write_soft(2)
code, out = run_update(answer="wet")
check(code == 0 and "REJECTED student_u2" in out, "update 2: a worse student is rejected")
with open(os.path.join(W, "deploy", "current", "info.json"), encoding="utf-8") as f:
    check(json.load(f)["version"] == "student_u1", "after a rejection the bin keeps student_u1")
check(os.path.isdir(os.path.join(W, "deploy", "previous")) is False, "no rollback copy made by a rejection")
with open(os.path.join(W, "models.csv"), newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
check([(r["version"], r["status"]) for r in rows] == [("student_u1", "deployed"), ("student_u2", "rejected")],
      "models.csv records the deployed and the rejected update")
check(rows[0]["threshold"] == "0.83" and rows[1]["threshold_source"] == "fallback 0.6", "threshold and its source recorded")
check(rows[1]["reasons"] != "" and rows[0]["test_fixed_errors"] == "2", "rejection reasons and test errors recorded")
check(Pool(rpool.root).check() == [], "pool rules still hold")

shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}")
sys.exit(1 if failures else 0)
