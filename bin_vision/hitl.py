"""
hitl.py
-------
The bin side of human-in-the-loop: when to ask a human, how to ask, and what
is saved for the teacher.

  ReviewPolicy     decides whether a load goes to a human:
                     "low_confidence"  assess() returned ASK_HUMAN
                     "audit"           every Xth load the bin would have tilted on
                                       its own is checked anyway. Audits are the
                                       unbiased measure of the bin's real accuracy,
                                       so a fall in human requests over time can be
                                       shown to be real and not overconfidence.
  KeyboardAsker    asks in an OpenCV window: shows each item's crop, d / w / x.
                   Any object with ask(crops) -> list of "dry" | "wet" | "" works,
                   so a button or screen version can replace it later.
  resolve()        what the bin does once the human has answered.
  FeedbackStore    saves what the teacher and the paper need:
                     loads.csv     one row per load, reviewed or not (the
                                   human-intervention rate over time is computed
                                   from this)
                     feedback.csv  one row per human-labelled item
                     crops/        the crop the student saw, for each labelled item
                                   (also the crop the human is shown, via crop_for)
                     frames/       the full camera frame of each reviewed load

Crops: an item labelled by the DropCounter when it landed carries that crop
(item.crop); it is shown, saved and marked from_drop = 1, because cutting it
again from the final frame could include an item that landed on top later.
Other items are cut from the final frame (from_drop = 0), which is also the
frame the student classified them in.

Loads that end MIXED or SEPARATE ("please separate") are logged in loads.csv but
produce no labels: the user splits them and each item is then sorted on its own.
"""

import csv
import os
import random
from dataclasses import dataclass, field
from datetime import datetime

import cv2

from multi_item import item_crop

LOAD_FIELDS = ["load_id", "timestamp", "student_version", "n_items", "composition", "load_confidence",
               "decision", "trigger", "human_labels", "final_action"]
FEEDBACK_FIELDS = ["crop_id", "load_id", "timestamp", "student_version", "item_index", "box",
                   "student_label", "student_confidence", "load_composition", "decision", "trigger",
                   "human_label", "crop_file", "from_drop"]


def crop_for(frame_bgr, item):
    """(crop, from_drop): the crop the student classified this item from. Items labelled
    when they landed keep that crop; others are cut from frame_bgr, the frame they were
    classified in."""
    if getattr(item, "crop", None) is not None:
        return item.crop, 1
    return item_crop(frame_bgr, item), 0


@dataclass
class ReviewPolicy:
    """audit_every = X. mode "every": exactly every Xth auto-tilt load; mode
    "random": each auto-tilt load with probability 1/X (unpredictable to users)."""
    audit_every: int = 20
    mode: str = "every"
    seed: int = 0
    auto_loads: int = field(default=0, init=False)

    def __post_init__(self):
        if self.audit_every < 1:
            raise ValueError("audit_every must be at least 1")
        self._rng = random.Random(self.seed)

    def review(self, decision):
        """Trigger name if this load must go to a human, else None. Call once per load."""
        if decision.action == "ASK_HUMAN":
            return "low_confidence"
        if decision.action.startswith("TILT"):
            self.auto_loads += 1
            if self.mode == "every" and self.auto_loads % self.audit_every == 0:
                return "audit"
            if self.mode == "random" and self._rng.random() < 1 / self.audit_every:
                return "audit"
        return None


def resolve(human_labels):
    """Bin action after the human answered, one label per item ("" = skipped)."""
    if not human_labels or any(l not in ("dry", "wet") for l in human_labels):
        return "REMOVE"                     # someone skipped an item: hand the load back
    kinds = set(human_labels)
    return f"TILT:{kinds.pop().upper()}" if len(kinds) == 1 else "SEPARATE"


class KeyboardAsker:
    """Asks for each item in an OpenCV window: d = dry, w = wet, x = skip."""

    def __init__(self, window="human review", size=360):
        self.window = window
        self.size = size

    def ask(self, crops, reason=""):
        labels = []
        for i, crop in enumerate(crops, start=1):
            h, w = crop.shape[:2]
            scale = self.size / max(h, w)
            view = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))))
            view = cv2.copyMakeBorder(view, 56, 0, 0, max(0, 420 - view.shape[1]), cv2.BORDER_CONSTANT, value=(40, 40, 40))
            cv2.putText(view, f"item {i}/{len(crops)}  {reason}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(view, "[d] dry   [w] wet   [x] skip", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow(self.window, view)
            key = 0
            while key not in (ord("d"), ord("w"), ord("x")):
                key = cv2.waitKey(0) & 0xFF
            labels.append({ord("d"): "dry", ord("w"): "wet", ord("x"): ""}[key])
        cv2.destroyWindow(self.window)
        return labels


class FeedbackStore:
    """Appends to loads.csv and feedback.csv under root; saves crops and frames."""

    def __init__(self, root, student_version):
        self.root = root
        self.student_version = student_version
        for sub in ("crops", "frames"):
            os.makedirs(os.path.join(root, sub), exist_ok=True)
        self.loads_csv = os.path.join(root, "loads.csv")
        self.feedback_csv = os.path.join(root, "feedback.csv")
        self.n_loads = self._count(self.loads_csv)

    @staticmethod
    def _count(path):
        if not os.path.exists(path):
            return 0
        with open(path, newline="", encoding="utf-8") as f:
            return sum(1 for _ in csv.DictReader(f))

    @staticmethod
    def _append(path, fields, rows):
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if new:
                writer.writeheader()
            writer.writerows(rows)

    def record(self, frame_bgr, decision, trigger=None, human_labels=None, final_action=None):
        """Log one load. If a human labelled it, also save every item's crop and label.
        Returns the load_id."""
        self.n_loads += 1
        now = datetime.now()
        load_id = f"{now:%Y%m%d-%H%M%S}-{self.n_loads:05d}"
        items = decision.items
        final_action = final_action or decision.action
        self._append(self.loads_csv, LOAD_FIELDS, [{
            "load_id": load_id, "timestamp": now.isoformat(timespec="seconds"), "student_version": self.student_version,
            "n_items": len(items), "composition": decision.composition, "load_confidence": f"{decision.load_confidence:.4f}",
            "decision": decision.action, "trigger": trigger or "", "human_labels": "+".join(human_labels or []),
            "final_action": final_action}])

        if human_labels:
            cv2.imwrite(os.path.join(self.root, "frames", f"{load_id}.jpg"), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows = []
            for i, (item, human) in enumerate(zip(items, human_labels), start=1):
                crop_id = f"{load_id}-{i}"
                crop_file = f"crops/{crop_id}.jpg"
                crop, from_drop = crop_for(frame_bgr, item)
                cv2.imwrite(os.path.join(self.root, crop_file), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({"crop_id": crop_id, "load_id": load_id, "timestamp": now.isoformat(timespec="seconds"),
                             "student_version": self.student_version, "item_index": i, "box": " ".join(map(str, item.box)),
                             "student_label": item.label, "student_confidence": f"{item.confidence:.4f}",
                             "load_composition": decision.composition, "decision": decision.action,
                             "trigger": trigger or "", "human_label": human, "crop_file": crop_file,
                             "from_drop": from_drop})
            self._append(self.feedback_csv, FEEDBACK_FIELDS, rows)
        return load_id


def summarize(root, window=50):
    """Human-request rate and audit accuracy per block of `window` loads, from loads.csv
    and feedback.csv: the main numbers for "human intervention decreases over time"."""
    with open(os.path.join(root, "loads.csv"), newline="", encoding="utf-8") as f:
        loads = list(csv.DictReader(f))
    audits = {}
    if os.path.exists(os.path.join(root, "feedback.csv")):
        with open(os.path.join(root, "feedback.csv"), newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["trigger"] == "audit" and r["human_label"]:
                    audits.setdefault(r["load_id"], []).append(r["student_label"] == r["human_label"])
    out = []
    for start in range(0, len(loads), window):
        block = loads[start:start + window]
        asked = sum(r["trigger"] == "low_confidence" for r in block)
        audited = [all(audits[r["load_id"]]) for r in block if r["load_id"] in audits]
        out.append({"loads": f"{start + 1}-{start + len(block)}",
                    "human_requests": asked, "request_rate": asked / len(block),
                    "audits": len(audited),
                    "audit_accuracy": (sum(audited) / len(audited)) if audited else None})
    return out
