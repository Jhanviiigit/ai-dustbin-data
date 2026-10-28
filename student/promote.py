"""Release gate: ship new weights only if they are calibrated and at least as good.

  python student/promote.py --candidate student/runs/new/weights.pt \
         --current student/releases/v1/weights.pt --data_dir dataset_bin

Both weights are scored on the same bin photos (dataset_bin/test, already crops). The
candidate is accepted when
  * it was calibrated (calibrate.py), and has a version name different from the current one,
  * its test accuracy is >= the current weights' accuracy.
On acceptance the weights are copied to <out>/<version>/weights.pt with an info.json. With
no --current (the first release) only the calibration check applies. Exit code 0 = accepted,
1 = rejected. --dry_run prints the verdict without copying.

Accuracy is the deciding number, as agreed; NLL, ECE and the paired counts are printed so a
tie or a one-image difference on a small test set is visible, not hidden.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from evaluate import metrics

p = argparse.ArgumentParser()
p.add_argument("--candidate", required=True)
p.add_argument("--current", default=None, help="weights currently on the bin (omit for the first release)")
p.add_argument("--data_dir", required=True, help="dataset_bin (its test/ split decides)")
p.add_argument("--out", default=str(Path(__file__).parent / "releases"))
p.add_argument("--dry_run", action="store_true")
args = p.parse_args()


def fail(msg):
    print(f"REJECTED: {msg}")
    sys.exit(1)


cand = metrics(args.candidate, args.data_dir, "test", no_crop=True)
ck = cand["ckpt"]
version = ck.get("version")
if not version:
    fail("the candidate has no version name (train with --version)")
if not ck.get("calibrated"):
    fail("the candidate is not calibrated: run calibrate.py on it first")
print(f"candidate {version}: test acc {cand['acc']:.4f}  NLL {cand['nll']:.4f}  ECE {cand['ece']:.4f}  "
      f"({cand['n']} images, T={cand['T']:.3f}, calibrated on {ck.get('calibrated_on')})")
if "bin" not in str(ck.get("calibrated_on", "")).lower():
    print("note: T was not fitted on bin photos, so 'confidence 0.8' may not mean 80% on the bin")

if args.current:
    cur = metrics(args.current, args.data_dir, "test", no_crop=True)
    cur_version = cur["ckpt"].get("version")
    if cur_version == version:
        fail(f"version name '{version}' is already the current one: give the new weights a new name")
    right_c, right_o = cand["preds"] == cand["labels"], cur["preds"] == cur["labels"]
    print(f"current   {cur_version}: test acc {cur['acc']:.4f}  NLL {cur['nll']:.4f}  ECE {cur['ece']:.4f}")
    print(f"paired: candidate right & current wrong {int((right_c & ~right_o).sum())}, "
          f"current right & candidate wrong {int((~right_c & right_o).sum())}")
    if cand["acc"] < cur["acc"]:
        fail(f"test accuracy {cand['acc']:.4f} is below the current {cur['acc']:.4f}")
else:
    cur_version = None
    print("no --current: first release")

dest = Path(args.out) / version
if dest.exists() and not args.dry_run:
    fail(f"{dest} already exists")
info = {"version": version, "test_accuracy": cand["acc"], "test_nll": cand["nll"], "test_ece": cand["ece"],
        "test_images": cand["n"], "temperature": cand["T"], "calibrated_on": ck.get("calibrated_on"),
        "replaces": cur_version, "init_from": ck.get("init_from"), "teacher_version": ck.get("teacher_version"),
        "feedback_used": ck.get("feedback_used"), "img_size": ck.get("img_size"), "seed": ck.get("seed")}
if args.dry_run:
    print("ACCEPTED (dry run, nothing copied)")
else:
    dest.mkdir(parents=True)
    shutil.copy2(args.candidate, dest / "weights.pt")
    (dest / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"ACCEPTED: released as {dest / 'weights.pt'}")
