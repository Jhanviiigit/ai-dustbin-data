"""
trigger.py
----------
Whether there is enough new training data for an update. Counted on the plan
AFTER the split, in GROUPS (not images), and only groups going to train: audits,
test_hard, val picks and augmented copies never count.

  update if   new train groups >= max(15, 15% of the train groups already in the pool)
              and >= 3 new train groups contain each class
  or          >= 14 days since the last update and >= 10 new train groups
              (no per-class minimum, so a rarely labelled class can't stall the loop;
              every retrain replays all old data, so a lopsided batch is safe)
"""

import math
from dataclasses import dataclass


@dataclass
class TriggerRule:
    k_min: int = 15
    fraction: float = 0.15
    per_class: int = 3
    days: float = 14
    time_min_groups: int = 10


def decide(plan, pool, ledger, rule=TriggerRule(), now=None):
    """(go, reason, counts)."""
    new_train = {}
    for p in plan:
        if p["split"] == "train" and pool.split_of_group(p["group_id"]) is None:
            new_train.setdefault(p["group_id"], set()).add(p["label"])
    n = len(new_train)
    per_class = {lab: sum(lab in labs for labs in new_train.values()) for lab in ("dry", "wet")}
    existing = len({r["group_id"] for r in pool.rows if r["split"] == "train" and r["origin"] != "augmented"})
    need = max(rule.k_min, math.ceil(rule.fraction * existing))
    days = ledger.since_last_update(now)
    counts = {"new_train_groups": n, "per_class": per_class, "needed": need, "days_since_update": round(days, 1)}

    if n >= need and min(per_class.values()) >= rule.per_class:
        return True, f"{n} new train groups (need {need}), {per_class}", counts
    if days >= rule.days and n >= rule.time_min_groups:
        return True, f"time trigger: {days:.0f} days since the last update, {n} new train groups", counts
    why = []
    if n < need:
        why.append(f"{n} of {need} new train groups")
    short = [lab for lab, c in per_class.items() if c < rule.per_class]
    if short:
        why.append(f"fewer than {rule.per_class} groups of {', '.join(short)} {per_class}")
    return False, "waiting: " + "; ".join(why), counts
