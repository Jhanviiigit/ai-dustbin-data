"""
collect.py
----------
New human-labelled feedback rows that are not in the pool yet.

Which rows are usable is decided by student/common.py load_feedback(), so the
pool, the teacher and the student apply exactly the same rules: skipped items,
missing files and crops cut from the final frame whose box overlaps another
item (from_drop = 0) are left out; drop-time crops (from_drop = 1) are kept.
"""

import contextlib
import csv
import io
import os
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from student.common import load_feedback


def collect(feedback_dirs, ledger):
    """(items, skipped): items are dicts with crop_id, src, label, load_id, trigger,
    from_drop, student_conf, feedback_dir, for rows the ledger hasn't processed."""
    items, skipped = [], Counter()
    for fdir in feedback_dirs:
        fdir = Path(fdir)
        with open(fdir / "feedback.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        with contextlib.redirect_stdout(io.StringIO()):          # load_feedback prints its own summary
            usable = {str(Path(p).resolve()) for p, _ in load_feedback(fdir)}
        for r in rows:
            if r["crop_id"] in ledger.processed:
                skipped["already in the pool"] += 1
                continue
            src = (fdir / r["crop_file"]).resolve()
            if str(src) not in usable:
                skipped["not usable (no label, missing, or overlapping final-frame crop)"] += 1
                continue
            items.append({"crop_id": r["crop_id"], "src": str(src), "label": r["human_label"],
                          "load_id": r["load_id"], "trigger": r.get("trigger", ""),
                          "from_drop": r.get("from_drop", ""), "student_conf": r.get("student_confidence", ""),
                          "feedback_dir": str(fdir)})
    return items, skipped
