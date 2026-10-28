"""Release gate: ship new weights only if they are calibrated and not worse than what shipped.

  python student/promote.py --candidate student/runs/student_v2/weights.pt \
         --current student/releases/student_v1/weights.pt --pool pool
  python student/promote.py --candidate ... --current ... --data_dir dataset_bin   # no pool yet

Every weights file is scored on the same images, so the comparison is paired:
  pool:      test_fixed (gated), test_audit (gated once it has >= 30 images), test_hard (reported only)
  data_dir:  its test/ split (gated)

The candidate is accepted when
  * it was calibrated (calibrate.py), and its version name is new (not current, not in --releases);
  * on every gated set, it makes at most MAX_EXTRA_ERRORS (2) more errors than the current
    weights AND than the best version ever shipped (every <releases>/*/weights.pt, re-scored
    now on the same images). Comparing with the best shipped stops slow drift: many updates
    that are each 2 errors worse cannot add up.
With no --current and no releases (the first release) only the calibration check applies.

On acceptance the weights are copied to <releases>/<version>/weights.pt with an info.json.
--report writes the full result as JSON (accepted or not; run_update.py puts it in models.csv).
Exit code 0 = accepted, 1 = rejected. --dry_run prints the verdict without copying.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from common import CLASSES, IMG_SIZE, LABELS, collect_logits, eval_tf, load_checkpoint
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "update"))
from metrics import describe, paired, summary  # noqa: E402

MAX_EXTRA_ERRORS = 2
AUDIT_MIN = 30


class Items(torch.utils.data.Dataset):
    """[(path, label)] -> (tensor, label); the images are already the detector's crops."""

    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        with Image.open(path) as im:
            return self.tf(im.convert("RGB")), label


def test_sets(pool=None, data_dir=None):
    """{name: (items, gated)} in a fixed order, so every model sees the same images."""
    if pool:
        from pool import Pool
        p = Pool(pool)
        sets = {}
        for split in ("test_fixed", "test_audit", "test_hard"):
            items = [(p.abs_path(r), LABELS.index(r["label"])) for r in p.rows_in(split)]
            gated = split == "test_fixed" or (split == "test_audit" and len(items) >= AUDIT_MIN)
            sets[split] = (items, gated)
        return sets
    items = []
    for label, cls in enumerate(CLASSES):
        folder = Path(data_dir) / "test" / cls
        items += [(f, label) for f in sorted(folder.iterdir()) if f.is_file()]
    return {"test": (items, True)}


def score(weights, sets):
    """{set name: (preds, labels)} and the checkpoint."""
    model, ckpt = load_checkpoint(weights)
    tf = eval_tf(ckpt.get("img_size", IMG_SIZE))
    out = {}
    for name, (items, _) in sets.items():
        if items:
            logits, labels = collect_logits(model, Items(items, tf))
            out[name] = (logits.argmax(1).numpy(), labels.numpy())
    return out, ckpt


def gate(candidate, sets, current=None, releases=None):
    """The verdict as a dict: accepted, reasons, and every number behind them."""
    cand, ck = score(candidate, sets)
    version = ck.get("version")
    report = {"version": version, "candidate": str(candidate), "calibrated": bool(ck.get("calibrated")),
              "temperature": ck.get("temperature"), "calibrated_on": ck.get("calibrated_on"),
              "max_extra_errors": MAX_EXTRA_ERRORS, "sets": {}, "reasons": []}

    shipped = {}                                   # version -> weights path
    for w in sorted(Path(releases).glob("*/weights.pt")) if releases and Path(releases).is_dir() else []:
        shipped[load_checkpoint(w)[1].get("version") or w.parent.name] = w
    cur_version = None
    if current:
        cur_version = load_checkpoint(current)[1].get("version")
        shipped.setdefault(cur_version, Path(current))
    report["current"], report["shipped"] = cur_version, sorted(shipped)

    if not version:
        report["reasons"].append("the candidate has no version name (train with --version)")
    elif version in shipped:
        report["reasons"].append(f"version name '{version}' was already shipped: give the new weights a new name")
    if not ck.get("calibrated"):
        report["reasons"].append("the candidate is not calibrated: run calibrate.py on it first")

    scores = {v: score(w, sets)[0] for v, w in shipped.items() if v != version}
    for name, (items, gated) in sets.items():
        entry = {"gated": gated, "n": len(items)}
        report["sets"][name] = entry
        if name not in cand:
            continue
        preds, labels = cand[name]
        entry["candidate"] = summary(labels, preds)
        if not scores:
            continue
        errors = {v: summary(labels, s[name][0])["errors"] for v, s in scores.items()}
        best = min(errors, key=lambda v: (errors[v], v != cur_version))
        entry["shipped_errors"] = errors
        entry["best"] = best
        entry["vs_best"] = paired(labels, preds, scores[best][name][0])
        if cur_version in scores:
            entry["vs_current"] = paired(labels, preds, scores[cur_version][name][0])
        # the best shipped has the fewest errors, so passing it also passes the current
        extra = entry["vs_best"]["extra_errors"]
        if gated and extra > MAX_EXTRA_ERRORS:
            who = "the current and best shipped" if best == cur_version else "the best shipped version"
            report["reasons"].append(f"{name}: {extra} more errors than {who} {best} (at most {MAX_EXTRA_ERRORS} allowed)")
    if not scores:
        report["note"] = "first release: nothing shipped to compare with, only the calibration check applies"
    report["accepted"] = not report["reasons"]
    return report


def print_report(r):
    print(f"candidate {r['version']}  (T={r['temperature']}, calibrated on {r['calibrated_on']})")
    print(f"current {r['current']}, shipped {r['shipped'] or 'none'}")
    for name, e in r["sets"].items():
        tag = "gated" if e["gated"] else ("reported" if name != "test_audit" else f"reported (gated from {AUDIT_MIN} images)")
        if "candidate" not in e:
            print(f"  {name:<10} empty")
            continue
        print(f"  {name:<10} [{tag}] {describe(e['candidate'])}")
        for key in ("vs_current", "vs_best"):
            if key in e:
                p = e[key]
                other = r["current"] if key == "vs_current" else e["best"]
                print(f"    {key:<10} {other}: only candidate wrong {p['only_a_wrong']}, only {other} wrong "
                      f"{p['only_b_wrong']}, extra errors {p['extra_errors']:+d} (McNemar p {p['mcnemar_p']:.3f})")
    if "note" in r:
        print(f"note: {r['note']}")
    if r["calibrated_on"] and "bin" not in str(r["calibrated_on"]).lower() and "pool" not in str(r["calibrated_on"]).lower():
        print("note: T was not fitted on bin photos, so 'confidence 0.8' may not mean 80% on the bin")


def _plain(x):
    """JSON-safe copy (tuples -> lists, NaN -> None)."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, float) and x != x:
        return None
    return x


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--current", default=None, help="weights currently on the bin (omit for the first release)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pool", help="the shared pool (test_fixed, test_audit, test_hard)")
    src.add_argument("--data_dir", help="dataset_bin (its test/ split decides)")
    p.add_argument("--releases", default=str(Path(__file__).parent / "releases"),
                   help="every shipped version, <releases>/<version>/weights.pt")
    p.add_argument("--report", default=None, help="write the result as JSON here")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    r = gate(args.candidate, test_sets(args.pool, args.data_dir), args.current, args.releases)
    print_report(r)
    dest = Path(args.releases) / str(r["version"])
    if r["accepted"] and dest.exists() and not args.dry_run:
        r["accepted"], r["reasons"] = False, [f"{dest} already exists"]
    r["date"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if args.report:
        Path(args.report).write_text(json.dumps(_plain(r), indent=1), encoding="utf-8")
    if not r["accepted"]:
        print("REJECTED: " + "; ".join(r["reasons"]))
        sys.exit(1)
    ck = load_checkpoint(args.candidate)[1]
    info = {**{k: r[k] for k in ("version", "date", "temperature", "calibrated_on")}, "replaces": r["current"],
            "init_from": ck.get("init_from"), "teacher_version": ck.get("teacher_version"),
            "feedback_used": ck.get("feedback_used"), "img_size": ck.get("img_size"), "seed": ck.get("seed"),
            "gate": _plain(r["sets"])}
    if args.dry_run:
        print("ACCEPTED (dry run, nothing copied)")
    else:
        dest.mkdir(parents=True)
        shutil.copy2(args.candidate, dest / "weights.pt")
        (dest / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"ACCEPTED: released as {dest / 'weights.pt'}")
