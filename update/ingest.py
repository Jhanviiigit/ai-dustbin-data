"""
ingest.py
---------
The data half of an update: new feedback -> split -> trigger -> pool.

  1. collect   new feedback rows not yet in the pool (student's feedback rules)
  2. split     by group and hash, with near-copy checks against the whole pool
  3. trigger   enough new train groups? (otherwise nothing is written; the rows stay pending)
  4. commit    add every row to its split, augment train up to balance, mark the rows processed,
               and write pool/updates/update_N.json (what went where and why)

Dry run by default: prints the plan and the trigger decision, writes nothing.

  python update/ingest.py --pool pool --feedback feedback                 # dry run
  python update/ingest.py --pool pool --feedback feedback --commit        # do it
  python update/ingest.py --pool pool --feedback feedback --commit --force   # ignore the trigger
  python update/ingest.py --pool pool --public_index public_phash.csv     # check public data only

The model half (teacher, student, gate, deploy, models.csv) comes in run_update.py.
"""

import argparse
import csv
import json
import os
from datetime import datetime

from augment import augment_to_balance
from collect import collect
from ledger import Ledger
from pool import Pool
from split import assign, public_leaks, summarize
from trigger import TriggerRule, decide


def ingest(pool_dir, feedback_dirs, commit=False, force=False, rule=TriggerRule(), now=None):
    """Returns a report dict; writes to the pool only if commit and (triggered or force)."""
    pool, ledger = Pool(pool_dir), Ledger(pool_dir)
    items, skipped = collect(feedback_dirs, ledger)
    plan = assign(items, pool)
    go, reason, counts = decide(plan, pool, ledger, rule, now)
    report = {"time": (now or datetime.now()).isoformat(timespec="seconds"), "new_rows": len(items),
              "skipped": dict(skipped), "plan": summarize(plan), "trigger": {"go": go, "reason": reason, **counts},
              "committed": False}
    if not (commit and (go or force)) or not plan:
        return report, plan

    number = ledger.updates + 1
    added_in = f"update_{number}"
    for p in plan:
        pool.add(p["src"], p["crop_id"], p["split"], p["label"], p["group_id"], "feedback",
                 trigger=p["trigger"], from_drop=p["from_drop"], added_in=added_in, student_conf=p["student_conf"])
    augmented = augment_to_balance(pool, added_in)
    problems = pool.check()
    if problems:
        raise RuntimeError("pool rules broken, nothing saved:\n" + "\n".join(problems[:20]))
    pool.save()
    ledger.commit([p["crop_id"] for p in plan], now)
    report.update(committed=True, update=added_in, augmented=len(augmented), forced=force and not go)
    os.makedirs(os.path.join(pool_dir, "updates"), exist_ok=True)
    with open(os.path.join(pool_dir, "updates", f"{added_in}.json"), "w", encoding="utf-8") as f:
        json.dump({**report, "rows": [{k: p[k] for k in ("crop_id", "split", "group_id", "label", "trigger", "reason",
                                                         "audit_leak")} for p in plan]}, f, indent=1)
    return report, plan


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="New feedback -> split -> trigger -> pool")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--feedback", nargs="*", default=[], help="hitl.py feedback folder(s)")
    ap.add_argument("--commit", action="store_true", help="write to the pool (default: dry run)")
    ap.add_argument("--force", action="store_true", help="commit even if the trigger says wait")
    ap.add_argument("--public_index", help="CSV (path, phash) of public images: list those that near-copy a test image")
    a = ap.parse_args()

    if a.public_index:
        leaks = public_leaks(Pool(a.pool), a.public_index)
        out = os.path.join(a.pool, "public_exclude.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "phash", "test_crop_id", "test_split", "distance"])
            w.writeheader()
            w.writerows(leaks)
        print(f"{len(leaks)} public images near-copy a pool test image -> {out} (the teacher leaves these out)")
    if a.feedback:
        report, plan = ingest(a.pool, a.feedback, a.commit, a.force)
        print(json.dumps(report, indent=1))
        if not a.commit:
            print("dry run: nothing written (add --commit)")
