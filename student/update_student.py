"""One update of the student: current weights + soft labels + feedback -> a calibrated new student.

  python student/update_student.py --current student/releases/student_v1/weights.pt \\
         --version student_v2 --data_root . \\
         --feedback_dir bin_feedback --soft_labels teacher/soft_labels_v2.csv

What it does (Package B of the update cycle):
  1. train.py --variant bin: starts from --current, trains on dataset_bin/train plus the
     human-labelled feedback crops, distilling from the teacher's soft labels
     (images without a teacher probability use their true label only).
  2. calibrate.py: fits the temperature on dataset_bin/val.
  3. writes <out>/update.json (what went in, val numbers, temperature) for the update
     script's models.csv registry.

Output <out>/weights.pt has the version name and the temperature inside. It does NOT ship
anything: decide that with promote.py (bin test set), which is the next command printed.
Exit code 0 = a calibrated new student was produced, non-zero = it failed (nothing to ship).

Rules kept here: dataset_bin/test is never read; feedback crops identical to a val/test image
are dropped; the new weights are calibrated before they are reported as done.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import VARIANTS, load_checkpoint
from evaluate import metrics

HERE = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--current", required=True, help="weights.pt of the student on the bin now (training starts from it)")
p.add_argument("--version", required=True, help="name of the new student, e.g. student_v2 (must differ from the current)")
p.add_argument("--data_root", default=".", help="folder that contains dataset_bin")
p.add_argument("--soft_labels", default=None, help="teacher soft-label CSV (Package A); omit for a plain fine-tune")
p.add_argument("--feedback_dir", default=None, help="hitl.py feedback folder (feedback.csv + crops/)")
p.add_argument("--out", default=None, help="default: student/runs/<version>")
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--lr", type=float, default=3e-5, help="fine-tuning rate; a starting guess, not tuned")
p.add_argument("--distill_alpha", type=float, default=0.5, help="weight of the true-label loss")
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--force", action="store_true", help="overwrite an existing <out>/weights.pt")
args = p.parse_args()


def fail(msg):
    print(f"FAILED: {msg}")
    sys.exit(1)


current = Path(args.current)
if not current.is_file():
    fail(f"--current not found: {current}")
_, cur = load_checkpoint(current)
if args.version == cur.get("version"):
    fail(f"version '{args.version}' is the current student's name: give the new one a new name")
bin_dir = Path(args.data_root) / VARIANTS["bin"]
if not (bin_dir / "val").is_dir():
    fail(f"{bin_dir}/val not found (dataset_bin from split_bin_dataset.py)")
for name, path in (("--soft_labels", args.soft_labels), ("--feedback_dir", args.feedback_dir)):
    if path and not Path(path).exists():
        fail(f"{name} not found: {path}")
if args.feedback_dir and not (Path(args.feedback_dir) / "feedback.csv").is_file():
    fail(f"{args.feedback_dir}/feedback.csv not found")
if not args.soft_labels and not args.feedback_dir:
    print("warning: no soft labels and no feedback: this only fine-tunes on dataset_bin again")
out = Path(args.out or HERE / "runs" / args.version)
if (out / "weights.pt").exists() and not args.force:
    fail(f"{out / 'weights.pt'} exists (use --force to replace it)")
out.mkdir(parents=True, exist_ok=True)


def run(script, *a):
    cmd = [sys.executable, str(HERE / script), *map(str, a)]
    print("\n$", " ".join(cmd), flush=True)
    with open(out / "update.log", "a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        if proc.wait():
            fail(f"{script} exited with code {proc.returncode} (see {out / 'update.log'})")


# 1. train from the current student on the bin photos + feedback, distilling from the teacher
train_args = ["--variant", "bin", "--data_root", args.data_root, "--init_weights", current, "--version", args.version,
              "--out", out, "--epochs", args.epochs, "--lr", args.lr, "--batch_size", args.batch_size,
              "--seed", args.seed, "--distill_alpha", args.distill_alpha]
if args.soft_labels:
    train_args += ["--soft_labels", args.soft_labels]
if args.feedback_dir:
    train_args += ["--feedback_dir", args.feedback_dir]
run("train.py", *train_args)

# 2. calibrate on dataset_bin/val (the default: the data it was trained on)
run("calibrate.py", "--weights", out / "weights.pt")

# 3. what went in and what came out
r = metrics(out / "weights.pt", bin_dir, "val", no_crop=True)
ck = r["ckpt"]
info = {"version": args.version, "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "replaces": cur.get("version"), "init_from": str(current),
        "train_images": ck.get("train_images"), "feedback_used": ck.get("feedback_used"),
        "feedback_dir": args.feedback_dir, "soft_labels": args.soft_labels, "soft_matched": ck.get("soft_matched"),
        "teacher_version": ck.get("teacher_version"), "distill_alpha": ck.get("distill_alpha"),
        "epochs": args.epochs, "lr": args.lr, "seed": args.seed, "img_size": ck.get("img_size"),
        "best_epoch": ck.get("epoch"), "val_accuracy": r["acc"], "val_nll": r["nll"], "val_ece": r["ece"],
        "val_images": r["n"], "temperature": r["T"], "calibrated": ck.get("calibrated"),
        "calibrated_on": ck.get("calibrated_on"), "weights": str(out / "weights.pt")}
(out / "update.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
if not ck.get("calibrated"):
    fail("the new weights are not calibrated")
print(f"\nDONE {args.version} (replaces {cur.get('version')}): val acc {r['acc']:.4f}, NLL {r['nll']:.4f}, "
      f"ECE {r['ece']:.4f}, T {r['T']:.3f}; trained on {info['train_images']} images "
      f"({info['feedback_used']} from feedback, {info['soft_matched']} with a teacher probability)")
print(f"-> {out / 'weights.pt'}   (details: {out / 'update.json'})")
print(f"Next, the gate:  python student/promote.py --candidate {out / 'weights.pt'} --current {current} "
      f"--data_dir {bin_dir}")
