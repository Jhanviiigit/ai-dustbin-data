"""Replay the loop-pile photos through the bin, one photo = one load, in shooting order:
detector (seg.py) -> assess (student + SEPARATE rule) -> ReviewPolicy -> FeedbackStore.
The person's answer is the folder label (dry / wet)."""
import csv, os, sys
import cv2
repo, piles, bgdir, cdir, deploy, feedback, maf = sys.argv[1:8]
sys.path.insert(0, os.path.join(repo, "bin_vision")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg import load_refs, segment
from multi_item import ShapeRule, DEFAULT_RULE, assess, make_classify, load_student
from hitl import FeedbackStore, ReviewPolicy, load_deployed, resolve
dep = load_deployed(deploy)
classify = make_classify(load_student(os.path.join(repo, "student"), dep["weights"]))
rule = ShapeRule(max_area_frac=float(maf), min_notch2=DEFAULT_RULE.min_notch2)
refs = load_refs(bgdir, 16)
store = FeedbackStore(feedback, dep["version"]); policy = ReviewPolicy().resume(store)
rows = [r for r in csv.DictReader(open(os.path.join(cdir, "crops.csv"), encoding="utf-8")) if r["pile"] == "loop"]
rows.sort(key=lambda r: r["original"])            # shooting order
log = []
for r in rows:
    frame = cv2.imread(os.path.join(piles, r["file"]))
    it, *_ = segment(refs, frame, r["ref"] or None)
    items = [it] if it else []
    if not items:
        log.append((r["item_id"], "NO_ITEM", "", "")); continue
    d = assess(frame, items, classify, dep["threshold"], rule)
    trig = policy.review(d)
    human = [r["label"]] * len(d.items) if trig else None
    store.record(frame, d, trig, human, resolve(human) if trig else None)
    log.append((r["item_id"], d.action, trig or "", r["label"]))
from collections import Counter
print("decisions", Counter(x[1] for x in log)); print("triggers", Counter(x[2] for x in log if x[2]))
wrong_auto = sum(1 for x in log if x[1].startswith("TILT") and not x[2] and x[1] != f"TILT:{x[3].upper()}")
print("auto-tilted wrongly (no person saw it):", wrong_auto)
