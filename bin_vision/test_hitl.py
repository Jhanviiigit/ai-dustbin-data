"""
test_hitl.py
------------
Checks the bin side of human-in-the-loop (hitl.py) on synthetic loads, with no
camera, model or window:

  - the student is a stand-in that reads the true labels from the scene and is
    unsure on a set share of items (as in test_composition.py)
  - the "human" is scripted and answers with the true labels

Checks:
  - ReviewPolicy: low-confidence loads go to a human; exactly every Xth auto-tilt
    load is audited ("every" mode); "random" mode audits about 1/X
  - resolve(): same class -> tilt, dry+wet -> separate, a skipped item -> remove
  - FeedbackStore: one loads.csv row per load; one feedback.csv row and one crop per
    human-labelled item; a frame only for reviewed loads; summarize() adds up

Usage:
  python test_hitl.py --clean ../../cleaned_v2 --masks ../../dataset_D_v2/masks
"""

import argparse
import csv
import os
import random
import sys
import tempfile

from hitl import FeedbackStore, ReviewPolicy, resolve, summarize
from multi_item import DEFAULT_RULE, Decision, DropCounter, ItemDetector, assess
from synth import ItemPool, SceneMaker, TRUTH_IDS

parser = argparse.ArgumentParser()
parser.add_argument("--clean", required=True)
parser.add_argument("--masks", required=True)
parser.add_argument("--loads", type=int, default=40)
parser.add_argument("--audit_every", type=int, default=5)
parser.add_argument("--unsure", type=float, default=0.15, help="Stand-in: share of items below the threshold")
args = parser.parse_args()

failures = []


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + what)
    if not cond:
        failures.append(what)


print("resolve():")
check(resolve(["dry", "dry"]) == "TILT:DRY", "dry+dry -> TILT:DRY")
check(resolve(["wet"]) == "TILT:WET", "wet -> TILT:WET")
check(resolve(["dry", "wet"]) == "SEPARATE", "dry+wet -> SEPARATE")
check(resolve(["dry", ""]) == "REMOVE", "a skipped item -> REMOVE")

print("ReviewPolicy:")
p = ReviewPolicy(audit_every=4)
tilt, ask = Decision("TILT:DRY", [], ""), Decision("ASK_HUMAN", [], "")
triggers = [p.review(tilt) for _ in range(12)]
check([i + 1 for i, t in enumerate(triggers) if t == "audit"] == [4, 8, 12], "every mode audits loads 4, 8, 12")
check(p.review(ask) == "low_confidence" and p.auto_loads == 12, "ASK_HUMAN -> low_confidence, not counted as an auto load")
check(p.review(Decision("MIXED", [], "")) is None, "MIXED is not reviewed")
r = ReviewPolicy(audit_every=10, mode="random", seed=1)
n = sum(r.review(tilt) == "audit" for _ in range(5000))
check(400 <= n <= 600, f"random mode audits about 1/10 ({n} of 5000)")

print("Loads through the store:")
pool = ItemPool(args.clean, args.masks, "test", seed=7)
maker = SceneMaker(pool, seed=7)
rng = random.Random(7)
root = tempfile.mkdtemp(prefix="hitl_test_")
store = FeedbackStore(root, "stand-in-v1")
policy = ReviewPolicy(audit_every=args.audit_every)
truth_ref = [None]
ID_LABEL = {v: k for k, v in TRUTH_IDS.items()}


def stand_in(frame, item):
    ids = truth_ref[0][item.mask]
    counts = {lab: int((ids == i).sum()) for lab, i in TRUTH_IDS.items()}
    label = max(counts, key=counts.get) if sum(counts.values()) else rng.choice(["dry", "wet"])
    return label, (rng.uniform(0.5, 0.59) if rng.random() < args.unsure else rng.uniform(0.8, 0.99))


expected = {"labelled_items": 0, "reviewed": 0, "audits": 0, "auto": 0}
for k in range(args.loads):
    labels = [rng.choice(["dry", "wet"])] if rng.random() < 0.7 else [rng.choice(["dry", "wet"])] * 2
    reference, frames, truths = maker.make_sequence(labels)
    detector = ItemDetector()
    detector.set_reference(reference)
    counter = DropCounter(detector)
    for frame, truth in zip(frames, truths):
        truth_ref[0] = truth
        counter.update(frame, stand_in)
    final = frames[-1]
    decision = assess(final, detector.detect(final), stand_in, 0.6, DEFAULT_RULE, counter.drop_items)
    trigger = policy.review(decision)
    if decision.action.startswith("TILT"):
        expected["auto"] += 1
    human = None
    if trigger:
        human = [ID_LABEL.get(int(truth_ref[0][it.mask].max()), "") for it in decision.items]   # scripted human
        expected["reviewed"] += 1
        expected["labelled_items"] += len(human)
        expected["audits"] += trigger == "audit"
    store.record(final, decision, trigger, human, resolve(human) if human else None)

loads = list(csv.DictReader(open(os.path.join(root, "loads.csv"), encoding="utf-8")))
feedback = list(csv.DictReader(open(os.path.join(root, "feedback.csv"), encoding="utf-8"))) \
    if os.path.exists(os.path.join(root, "feedback.csv")) else []
check(len(loads) == args.loads, f"loads.csv has one row per load ({len(loads)})")
check(len(feedback) == expected["labelled_items"], f"feedback.csv has one row per labelled item ({len(feedback)})")
check(all(os.path.exists(os.path.join(root, r["crop_file"])) for r in feedback), "every labelled item's crop exists")
frames_saved = len(os.listdir(os.path.join(root, "frames")))
check(frames_saved == expected["reviewed"], f"a frame is saved only for reviewed loads ({frames_saved})")
check(sum(r["trigger"] == "audit" for r in loads) == expected["audits"] == expected["auto"] // args.audit_every,
      f"audits = auto-tilt loads // X ({expected['audits']} of {expected['auto']})")
check(all(r["student_version"] == "stand-in-v1" for r in loads), "student version is logged on every load")
blocks = summarize(root, window=20)
check(sum(b["human_requests"] for b in blocks) == sum(r["trigger"] == "low_confidence" for r in loads),
      "summarize() counts every low-confidence request")
for b in blocks:
    print("   ", b)

print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}  (files in {root})")
sys.exit(1 if failures else 0)
