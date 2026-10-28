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
  - the agreed audit rate is 1 in 10; after a restart, resume() keeps "every" at every Xth
  - load_deployed(): deploy/current's weights, version and threshold (0.6 if none)
  - FeedbackStore: one loads.csv row per load; one feedback.csv row and one crop per
    human-labelled item; a frame only for reviewed loads; summarize() adds up

Usage:
  python test_hitl.py --clean ../../cleaned_v2 --masks ../../dataset_D_v2/masks
"""

import argparse
import csv
import json
import os
import random
import sys
import tempfile

from hitl import AUDIT_EVERY, FeedbackStore, ReviewPolicy, load_deployed, resolve, summarize
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
check(ReviewPolicy().audit_every == AUDIT_EVERY == 10, "default audit rate: 1 in 10")
rs = tempfile.mkdtemp(prefix="hitl_resume_")
st = FeedbackStore(rs, "v1")
first = ReviewPolicy(audit_every=10)
for _ in range(7):
    st.record(None, tilt, first.review(tilt))
st.record(None, ask, first.review(ask))
second = ReviewPolicy(audit_every=10).resume(st)
later = [second.review(tilt) for _ in range(13)]
check(second.auto_loads == 20 and [i for i, t in enumerate(later, start=8) if t == "audit"] == [10, 20],
      "after a restart after 7 auto loads, the audits are still loads 10 and 20")

print("Deployed student:")
dep = tempfile.mkdtemp(prefix="hitl_deploy_")
try:
    load_deployed(dep)
    check(False, "an empty deploy folder is refused")
except FileNotFoundError:
    check(True, "an empty deploy folder is refused")
os.makedirs(os.path.join(dep, "current"))
open(os.path.join(dep, "current", "weights.pt"), "wb").close()
info = {"version": "student_u3", "threshold": 0.83, "threshold_source": "val (update.json)"}
json.dump(info, open(os.path.join(dep, "current", "info.json"), "w"))
d = load_deployed(dep)
check(d["version"] == "student_u3" and d["threshold"] == 0.83 and d["weights"].endswith("weights.pt"),
      "version, threshold and weights from deploy/current (run_update.py's info.json)")
json.dump({"version": "student_u4", "threshold": None}, open(os.path.join(dep, "current", "info.json"), "w"))
d = load_deployed(dep)
check(d["threshold"] == 0.6 and d["threshold_source"] == "fallback 0.6", "no threshold deployed: 0.6")

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
check(all(r["from_drop"] in ("0", "1") for r in feedback), "feedback.csv has from_drop on every row")

print("Covered items keep the crop they were labelled from:")
import cv2
import numpy as np
from hitl import crop_for
from multi_item import item_crop

seen = {}                                  # id(item) -> the crop the student was given


def recording_stand_in(frame, item):
    seen[id(item)] = item_crop(frame, item)
    return stand_in(frame, item)


store2 = FeedbackStore(tempfile.mkdtemp(prefix="hitl_test_"), "stand-in-v1")
exact, shown_same, saved_same, flagged, would_differ, n_cases = 0, 0, 0, 0, 0, 0
for k in range(12):
    labels = [rng.choice(["dry", "wet"]), rng.choice(["dry", "wet"])]
    reference, frames, truths = maker.make_sequence(labels, touching=True)        # second item lands against the first
    detector = ItemDetector()
    detector.set_reference(reference)
    counter = DropCounter(detector)
    for frame, truth in zip(frames, truths):
        truth_ref[0] = truth
        counter.update(frame, recording_stand_in)
    final = frames[-1]
    decision = assess(final, detector.detect(final), recording_stand_in, 0.6, None, counter.drop_items)
    if len(counter.drop_items) < 2 or decision.items is not counter.drop_items and \
            list(decision.items) != list(counter.drop_items):
        continue                           # the counter missed a drop: not this case
    n_cases += 1
    first = counter.drop_items[0]
    exact += first.crop is not None and np.array_equal(first.crop, seen[id(first)])
    shown, from_drop = crop_for(final, first)
    shown_same += np.array_equal(shown, first.crop) and from_drop == 1
    would_differ += not np.array_equal(item_crop(final, first), first.crop)
    load_id = store2.record(final, decision, "audit", labels, resolve(labels))
    rows = [r for r in csv.DictReader(open(os.path.join(store2.root, "feedback.csv"), encoding="utf-8"))
            if r["load_id"] == load_id]
    saved = cv2.imread(os.path.join(store2.root, rows[0]["crop_file"]))
    saved_same += saved.shape == first.crop.shape and np.abs(saved.astype(int) - first.crop.astype(int)).mean() < 3
    flagged += rows[0]["from_drop"] == "1"
check(n_cases >= 6, f"{n_cases} of 12 touching loads were labelled at drop time")
check(exact == n_cases, "each item keeps exactly the crop the student classified")
check(shown_same == n_cases, "the human is shown that same crop (from_drop = 1)")
check(saved_same == n_cases, "the saved crop is that crop (up to JPEG)")
check(flagged == n_cases, "feedback.csv marks it from_drop = 1")
check(would_differ > 0, f"re-cutting from the final frame would have changed {would_differ} of {n_cases} crops (the old bug)")

print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}  (files in {root})")
sys.exit(1 if failures else 0)
