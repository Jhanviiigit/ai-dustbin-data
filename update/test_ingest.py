"""
test_ingest.py
--------------
Tests for phase 2 (collect, split, trigger, augment, ingest) on generated images
and hand-made feedback folders. No datasets needed.

  python update/test_ingest.py
"""

import csv
import os
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augment import make_copy
from ingest import ingest
from ledger import Ledger
from pool import Pool, bucket, phash
from split import public_leaks
from trigger import TriggerRule, decide

failures = []
TMP = tempfile.mkdtemp(prefix="ingest_test_")


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + what)
    if not cond:
        failures.append(what)


def image(seed):
    """A random blocky picture: different seeds give pHashes ~30 bits apart."""
    r = np.random.default_rng(seed)
    small = r.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((96, 96), Image.NEAREST)


def make_pool(n_train=6):
    """A pool with a few capture images in train, val and test_fixed."""
    pool = Pool(os.path.join(TMP, "pool"))
    k = 0
    for split, n in (("train", n_train), ("val", 2), ("test_fixed", 2)):
        for i in range(n):
            k += 1
            pool.add(image(1000 + k), f"cap{k}", split, "dry" if i % 2 else "wet", f"item{k}", "capture", added_in="seed")
    pool.save()
    return pool


def make_feedback(name, loads):
    """loads: list of (load_id, trigger, [(label, image, box, from_drop), ...]) -> feedback folder."""
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


def expected_split(group, trigger):
    if trigger == "audit":
        return "test_audit"
    if bucket(group, 5) == 0:
        return "test_hard"
    if bucket(group, 100) < 15:
        return "val"
    return "train"


# ─── 1. Trigger rules (on plans, no images) ───────────────────────────────────
print("Trigger:")
pool = make_pool()
ledger = Ledger(pool.root)


def plan_of(dry, wet):
    return [{"split": "train", "group_id": f"d{i}", "label": "dry"} for i in range(dry)] + \
           [{"split": "train", "group_id": f"w{i}", "label": "wet"} for i in range(wet)]


now = datetime.now()
check(not decide(plan_of(15, 0), pool, ledger, now=now)[0], "15 dry + 0 wet groups: wait")
check(decide(plan_of(12, 3), pool, ledger, now=now)[0], "12 dry + 3 wet groups: update")
check(not decide(plan_of(2, 13), pool, ledger, now=now)[0], "2 dry + 13 wet groups: wait")
check(not decide(plan_of(10, 4), pool, ledger, now=now)[0], "14 groups: wait (need 15)")
others = [{"split": s, "group_id": f"x{i}", "label": "wet"} for i, s in enumerate(["test_audit", "test_hard", "val"] * 3)]
check(not decide(plan_of(11, 3) + others, pool, ledger, now=now)[0],
      "14 train groups + 9 audit/test_hard/val groups: wait (only train groups count)")
check(decide(plan_of(12, 3), pool, ledger, TriggerRule(k_min=15), now=now)[2]["needed"] == 15, "need = max(15, 15% of train groups)")
ledger.started = (now - timedelta(days=15)).isoformat(timespec="seconds")
check(decide(plan_of(10, 0), pool, ledger, now=now)[0], "time trigger: 15 days, 10 groups, one class: update")
check(not decide(plan_of(9, 0), pool, ledger, now=now)[0], "time trigger: 15 days, 9 groups: wait")
shutil.rmtree(pool.root)

# ─── 2. Routing, dry run, pending ─────────────────────────────────────────────
print("Routing and dry run:")
pool = make_pool()
loads = [(f"L{i:03d}", "low_confidence", [("dry" if i % 3 else "wet", image(i), "10 10 40 40", 1)]) for i in range(40)]
loads += [(f"A{i:03d}", "audit", [("wet", image(500 + i), "10 10 40 40", 1)]) for i in range(5)]
fb1 = make_feedback("fb1", loads)
before = len(Pool(pool.root).rows)
report, plan = ingest(pool.root, [fb1])
check(report["committed"] is False and len(Pool(pool.root).rows) == before, "dry run writes nothing")
check(not os.path.exists(os.path.join(pool.root, "ledger.json")), "dry run leaves no ledger")
check(all(p["split"] == expected_split(p["group_id"], p["trigger"]) for p in plan), "every group routed by the hash rules")
check(all(p["split"] == "test_audit" for p in plan if p["trigger"] == "audit"), "audits go to test_audit")
splits = Counter(p["split"] for p in plan)
check(splits["train"] > 0 and splits["test_hard"] + splits["val"] > 0, f"low-confidence groups spread over splits {dict(splits)}")
_, plan_again = ingest(pool.root, [fb1])
check([p["split"] for p in plan_again] == [p["split"] for p in plan], "same input, same plan (no arrival-order effects)")

print("Pending when the trigger says wait:")
small = make_feedback("fb_small", loads[:3])
report, _ = ingest(pool.root, [small], commit=True)
check(not report["committed"] and Ledger(pool.root).processed == set(), "not enough groups: nothing committed, rows stay pending")

# ─── 3. Commit, augmentation, permanence ──────────────────────────────────────
print("Commit and augmentation:")
report, plan = ingest(pool.root, [fb1], commit=True, force=True)
pool = Pool(pool.root)
check(report["committed"] and report["update"] == "update_1", "forced commit writes update_1")
check(os.path.exists(os.path.join(pool.root, "updates", "update_1.json")), "updates/update_1.json records the plan")
check(len(Ledger(pool.root).processed) == len(plan), "ledger marks every committed row processed")
check(pool.check() == [], "pool rules hold after commit")
train = pool.rows_in("train")
real = Counter(r["label"] for r in train if r["origin"] != "augmented")
total = Counter(r["label"] for r in train)
copies = Counter(r["parent_crop_id"] for r in train if r["origin"] == "augmented")
check(all(c <= 2 for c in copies.values()), f"at most 2 copies per original ({max(copies.values(), default=0)} max)")
cap_limited = min(real.values()) * 3 < max(real.values())
check(total["dry"] == total["wet"] or cap_limited, f"classes balanced up to the cap: real {dict(real)}, with copies {dict(total)}")
check(all(r["split"] == "train" for r in pool.rows if r["origin"] == "augmented"), "augmented copies only in train")
a = np.asarray(make_copy(image(7), "x_aug1"))
check(np.array_equal(a, np.asarray(make_copy(image(7), "x_aug1"))) and not np.array_equal(a, np.asarray(make_copy(image(7), "x_aug2"))),
      "copies are reproducible from their crop_id, and different per copy")
capped = Pool(os.path.join(TMP, "capped"))
for i in range(9):
    capped.add(image(4000 + i), f"d{i}", "train", "dry", f"gd{i}", "capture")
capped.add(image(4100), "w0", "train", "wet", "gw0", "capture")
from augment import augment_to_balance
added = augment_to_balance(capped, "t")
check(len(added) == 2 and Counter(r["label"] for r in capped.rows_in("train"))["wet"] == 3,
      "9 dry vs 1 wet: only 2 copies (cap), so 3 wet; class weights handle the rest")
again, _ = ingest(pool.root, [fb1], commit=True, force=True)
check(again["new_rows"] == 0, "committed rows are not collected again")

print("Splits are permanent:")
first = {p["group_id"]: p["split"] for p in plan}
fb2 = make_feedback("fb2", [(l, "low_confidence", [("dry", image(900 + i), "10 10 40 40", 1)])
                            for i, l in enumerate(list(first)[:6])])
_, plan2 = ingest(pool.root, [fb2])
check(all(p["split"] == first[p["group_id"]] and "already in the pool" in p["reason"] for p in plan2),
      "new images of a known group follow its split")

# ─── 4. Leak checks ───────────────────────────────────────────────────────────
print("Leak checks:")
test_row = pool.rows_in("test_fixed")[0]
train_row = next(r for r in pool.rows_in("train") if r["origin"] == "capture")
fb3 = make_feedback("fb3", [
    ("N001", "low_confidence", [(test_row["label"], Image.open(pool.abs_path(test_row)), "10 10 40 40", 1)]),
    ("N002", "audit", [(train_row["label"], Image.open(pool.abs_path(train_row)), "10 10 40 40", 1)]),
    ("N003", "low_confidence", [("dry", image(7001), "10 10 40 40", 1)]),
    ("N004", "low_confidence", [("dry", image(7001), "10 10 40 40", 1)]),
])
_, plan3 = ingest(pool.root, [fb3])
p = {x["load_id"]: x for x in plan3}
check(p["N001"]["split"] == "test_fixed" and p["N001"]["group_id"] == test_row["group_id"],
      "near-copy of a test image goes to that test split and group, not train")
check(p["N002"]["split"] == "train" and p["N002"]["audit_leak"], "audit near-copying a train image goes to train, flagged audit_leak")
check(p["N003"]["group_id"] == p["N004"]["group_id"] and p["N003"]["split"] == p["N004"]["split"],
      "near-copies within one batch share a group and split")

idx = os.path.join(TMP, "public.csv")
with open(idx, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["path", "phash"])
    w.writerow(["public/a.jpg", test_row["phash"]])
    w.writerow(["public/b.jpg", phash(image(8888))])
leaks = public_leaks(pool, idx)
check([x["path"] for x in leaks] == ["public/a.jpg"], "public image matching a test image is reported for exclusion")

# ─── 5. Feedback rules shared with the student ────────────────────────────────
print("Feedback rules (from student/common.py):")
fb4 = make_feedback("fb4", [
    ("O001", "low_confidence", [("dry", image(3001), "10 10 60 60", 0), ("wet", image(3002), "40 40 60 60", 0)]),
    ("O002", "low_confidence", [("dry", image(3003), "10 10 60 60", 1), ("wet", image(3004), "40 40 60 60", 1)]),
    ("O003", "low_confidence", [("", image(3005), "10 10 40 40", 1)]),
])
_, plan4 = ingest(pool.root, [fb4])
ids = sorted(x["crop_id"] for x in plan4)
check(ids == ["O002-1", "O002-2"], f"overlapping final-frame crops and unlabelled rows left out, drop-time crops kept ({ids})")

shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}")
sys.exit(1 if failures else 0)
