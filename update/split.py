"""
split.py
--------
Decides where each new feedback image goes, by GROUP and by HASH, never by
arrival order. Nothing is written here: assign() returns a plan.

For each new group (feedback load_id; near-copies within the batch merged):
  1. group already in the pool        -> its split (splits are permanent)
  2. any image is a near-copy (pHash distance <= 4) of any pool image
                                       -> the whole group joins that image's group and split.
     This blocks leaks both ways: a near-copy of a test image never reaches
     train, and an audit near-copying a train/val image goes there instead of
     test_audit (logged as "audit_leak", reported per update).
  3. trigger = audit                   -> test_audit
  4. low_confidence, bucket(group, 5) == 0   -> test_hard
  5. low_confidence, bucket(group, 100) < 15 -> val
  6. otherwise                         -> train
A merged group holding both audit and low_confidence loads is routed as
low_confidence and its audits are logged as audit_leak (rare).

public_leaks() checks public datasets against the pool's test images, so the
teacher can leave out any public image that near-copies a test image.
"""

import csv
from collections import Counter

import numpy as np
from PIL import Image, ImageOps

from pool import TEST_SPLITS, bucket, phash

NEAR = 4


def _hash_matrix(hex_hashes):
    return np.array([int(h, 16) for h in hex_hashes], dtype=np.uint64)


def _nearest(hash_hex, matrix):
    """(index, distance) of the closest hash in matrix, or (None, None) if empty."""
    if len(matrix) == 0:
        return None, None
    d = np.bitwise_count(matrix ^ np.uint64(int(hash_hex, 16)))
    i = int(d.argmin())
    return i, int(d[i])


def assign(items, pool, near=NEAR):
    """Plan for new feedback items: list of dicts (the item plus split, group_id, reason)."""
    for it in items:
        it["phash"] = phash(ImageOps.exif_transpose(Image.open(it["src"])))

    # Groups: load_id, then merge near-copies within the batch (union-find)
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_load = {}
    for i, it in enumerate(items):
        if it["load_id"] in by_load:
            parent[find(i)] = find(by_load[it["load_id"]])
        by_load.setdefault(it["load_id"], i)
    if items:
        m = _hash_matrix([it["phash"] for it in items])
        for i in range(len(items)):
            for j in np.nonzero(np.bitwise_count(m[i + 1:] ^ m[i]) <= near)[0]:
                parent[find(i + 1 + int(j))] = find(i)
    groups = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(items[i])

    pool_rows = [r for r in pool.rows if r["origin"] != "augmented"]
    pool_matrix = _hash_matrix([r["phash"] for r in pool_rows])
    plan = []
    for members in groups.values():
        loads = sorted({it["load_id"] for it in members})
        group_id = loads[0]
        triggers = {it["trigger"] for it in members}
        is_audit = triggers == {"audit"}
        split, reason = None, ""

        known = next((l for l in loads if pool.split_of_group(l)), None)
        if known:
            split, group_id, reason = pool.split_of_group(known), known, "group already in the pool"
        if split is None:
            for it in members:
                i, d = _nearest(it["phash"], pool_matrix)
                if i is not None and d <= near:
                    hit = pool_rows[i]
                    split, group_id = hit["split"], hit["group_id"]
                    reason = f"near-copy (distance {d}) of {hit['split']} image {hit['crop_id']}"
                    break
        if split is None:
            if is_audit:
                split, reason = "test_audit", "audit"
            elif bucket(group_id, 5) == 0:
                split, reason = "test_hard", "low_confidence, hash bucket 0 of 5"
            elif bucket(group_id, 100) < 15:
                split, reason = "val", "low_confidence, hash bucket < 15 of 100"
            else:
                split, reason = "train", "low_confidence"
        leak = "audit" in triggers and split not in TEST_SPLITS
        for it in members:
            plan.append({**it, "split": split, "group_id": group_id, "reason": reason,
                         "audit_leak": leak and it["trigger"] == "audit"})
    return plan


def summarize(plan):
    return {"by_split": dict(Counter(p["split"] for p in plan)),
            "audit_leaks": sum(p["audit_leak"] for p in plan),
            "near_copies": sum(p["reason"].startswith("near-copy") for p in plan)}


def public_leaks(pool, public_index_csv, near=NEAR):
    """Public images (rows of a CSV with path, phash) that near-copy a pool test image.
    The teacher must leave these out of training."""
    tests = pool.rows_in(*TEST_SPLITS)
    matrix = _hash_matrix([r["phash"] for r in tests])
    out = []
    with open(public_index_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            i, d = _nearest(r["phash"], matrix)
            if i is not None and d <= near:
                out.append({"path": r["path"], "phash": r["phash"], "test_crop_id": tests[i]["crop_id"],
                            "test_split": tests[i]["split"], "distance": d})
    return out
