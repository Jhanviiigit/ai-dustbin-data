"""
pool.py
-------
The shared pool: every bin-camera image the teacher and student train and are
tested on, in one folder with one manifest. Only the update pipeline writes it
(through this module); the teacher and student only read it.

Layout (the dataset_bin layout, with more splits):

  pool/
    train/      {dry waste,wet waste}/<crop_id>.jpg   capture + feedback + augmented copies
    val/        ...                                   picks settings (tau, threshold, calibration)
    test_fixed/ ...                                   dataset_bin/test, never changes; the gate's score
    test_audit/ ...                                   audit labels: accuracy of automatic decisions
    test_hard/  ...                                   some low-confidence groups; reported, never gated
    manifest.csv

manifest.csv columns (agreed format):
  crop_id         unique; the file name without .jpg
  path            relative to pool/, forward slashes
  split           train | val | test_fixed | test_audit | test_hard
  label           dry | wet               (folders are "dry waste" / "wet waste")
  group_id        capture: item_id; public: the dataset's group; feedback: load_id; augmented: its original's
  origin          capture (bin camera) | public (a starting dataset, e.g. Organik) | feedback | augmented
  parent_crop_id  augmented copies: the original's crop_id; otherwise empty
  trigger         feedback: low_confidence | audit; otherwise empty
  from_drop       feedback: 1 | 0 (hitl.py); otherwise empty
  phash           64-bit perceptual hash, 16 hex digits
  added_in        which update (or "seed") added the row
  student_conf    feedback: the student's confidence when it was labelled; otherwise empty

Rules enforced here:
  - crop_ids are unique; labels, splits, origins and triggers use the agreed values
  - a group's split never changes: every image of a group is in one split
  - test splits hold real images only (no augmented copies)
  - an augmented copy points to an existing original in train, in the same group

Command line:
  python update/pool.py seed  --pool pool --organik dataset_organik_v2     # the starting data
  python update/pool.py seed  --pool pool --dataset_bin dataset_bin       # bin-camera captures
  python update/pool.py check --pool pool
"""

import argparse
import csv
import os
import shutil
from collections import Counter

import numpy as np
from PIL import Image, ImageOps

SPLITS = ("train", "val", "test_fixed", "test_audit", "test_hard")
TEST_SPLITS = ("test_fixed", "test_audit", "test_hard")
LABELS = ("dry", "wet")
CLASS_FOLDERS = {"dry": "dry waste", "wet": "wet waste"}
ORIGINS = ("capture", "public", "feedback", "augmented")
TRIGGERS = ("low_confidence", "audit", "")
COLUMNS = ["crop_id", "path", "split", "label", "group_id", "origin", "parent_crop_id", "trigger",
           "from_drop", "phash", "added_in", "student_conf"]
DATASET_BIN_SPLITS = {"train": "train", "val": "val", "test": "test_fixed"}
FOLDER_LABELS = {"dry waste": "dry", "wet waste": "wet"}


# ─── Perceptual hash (same algorithm as data_prep/clean_dataset.py) ───────────
def _dct_matrix(n):
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    m[0] *= 1 / np.sqrt(2)
    return m * np.sqrt(2 / n)


_DCT32 = _dct_matrix(32)


def phash(image):
    """64-bit perceptual hash of a PIL image, as 16 hex digits."""
    pixels = np.asarray(image.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    low = (_DCT32 @ pixels @ _DCT32.T)[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def hamming(a, b):
    """Bits that differ between two hex hashes."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def bucket(group_id, n):
    """Stable 0..n-1 bucket for a group, the same on every machine and run (Python's
    built-in hash() changes between runs, so it can't be used for splits)."""
    import hashlib
    return int(hashlib.sha1(str(group_id).encode("utf-8")).hexdigest(), 16) % n


# ─── The pool ─────────────────────────────────────────────────────────────────
class PoolError(ValueError):
    pass


class Pool:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.manifest_path = os.path.join(self.root, "manifest.csv")
        self.rows = []
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, newline="", encoding="utf-8") as f:
                self.rows = [{c: r.get(c, "") for c in COLUMNS} for r in csv.DictReader(f)]
        self.by_id = {r["crop_id"]: r for r in self.rows}
        self.group_split = {}
        for r in self.rows:
            self.group_split.setdefault(r["group_id"], r["split"])

    # Reading -----------------------------------------------------------------
    def split_of_group(self, group_id):
        """The split a group was assigned to, or None if the group is new."""
        return self.group_split.get(group_id)

    def rows_in(self, *splits):
        return [r for r in self.rows if r["split"] in splits]

    def counts(self):
        return Counter((r["split"], r["label"]) for r in self.rows)

    def abs_path(self, row):
        return os.path.join(self.root, *row["path"].split("/"))

    # Writing -----------------------------------------------------------------
    def add(self, src, crop_id, split, label, group_id, origin, parent_crop_id="", trigger="",
            from_drop="", added_in="", student_conf=""):
        """Copy one image into the pool and add its manifest row. src is a file path or a PIL image.
        Raises PoolError (and changes nothing) if a rule would be broken."""
        crop_id, group_id = str(crop_id), str(group_id)
        if not crop_id or "/" in crop_id or "\\" in crop_id:
            raise PoolError(f"bad crop_id {crop_id!r}")
        if crop_id in self.by_id:
            raise PoolError(f"crop_id {crop_id} is already in the pool")
        if split not in SPLITS:
            raise PoolError(f"split {split!r} is not one of {SPLITS}")
        if label not in LABELS:
            raise PoolError(f"label {label!r} is not one of {LABELS}")
        if origin not in ORIGINS:
            raise PoolError(f"origin {origin!r} is not one of {ORIGINS}")
        if trigger not in TRIGGERS:
            raise PoolError(f"trigger {trigger!r} is not one of {TRIGGERS}")
        if not group_id:
            raise PoolError(f"{crop_id}: group_id is empty")
        known = self.group_split.get(group_id)
        if known is not None and known != split:
            raise PoolError(f"{crop_id}: group {group_id} is in {known}, cannot add it to {split} (splits are permanent)")
        if origin == "augmented":
            if split in TEST_SPLITS or split == "val":
                raise PoolError(f"{crop_id}: augmented copies go to train only, not {split}")
            parent = self.by_id.get(parent_crop_id)
            if parent is None:
                raise PoolError(f"{crop_id}: parent {parent_crop_id!r} is not in the pool")
            if parent["group_id"] != group_id or parent["label"] != label:
                raise PoolError(f"{crop_id}: must share its parent's group and label")
        elif parent_crop_id:
            raise PoolError(f"{crop_id}: only augmented copies have a parent_crop_id")

        image = src if isinstance(src, Image.Image) else ImageOps.exif_transpose(Image.open(src))
        image = image.convert("RGB")
        rel = f"{split}/{CLASS_FOLDERS[label]}/{crop_id}.jpg"
        dest = os.path.join(self.root, *rel.split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if isinstance(src, str) and src.lower().endswith((".jpg", ".jpeg")):
            shutil.copy2(src, dest)                      # keep the exact bytes of real crops
        else:
            image.save(dest, quality=95)
        row = {"crop_id": crop_id, "path": rel, "split": split, "label": label, "group_id": group_id,
               "origin": origin, "parent_crop_id": parent_crop_id, "trigger": trigger,
               "from_drop": str(from_drop), "phash": phash(image), "added_in": added_in,
               "student_conf": str(student_conf)}
        self.rows.append(row)
        self.by_id[crop_id] = row
        self.group_split.setdefault(group_id, split)
        return row

    def save(self):
        """Write manifest.csv (to a temporary file first, so a crash never leaves half a manifest)."""
        os.makedirs(self.root, exist_ok=True)
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(tmp, self.manifest_path)

    # Checking ----------------------------------------------------------------
    def check(self):
        """All rule violations in the pool as a list of strings (empty = fine)."""
        problems, seen, group_splits = [], set(), {}
        for r in self.rows:
            cid = r["crop_id"]
            if cid in seen:
                problems.append(f"duplicate crop_id {cid}")
            seen.add(cid)
            if r["split"] not in SPLITS or r["label"] not in LABELS or r["origin"] not in ORIGINS \
                    or r["trigger"] not in TRIGGERS:
                problems.append(f"{cid}: value outside the agreed set ({r['split']}, {r['label']}, {r['origin']}, {r['trigger']})")
            if "\\" in r["path"] or os.path.isabs(r["path"]):
                problems.append(f"{cid}: path must be relative with forward slashes: {r['path']}")
            elif r["split"] in SPLITS and r["label"] in LABELS and \
                    r["path"] != f"{r['split']}/{CLASS_FOLDERS[r['label']]}/{cid}.jpg":
                problems.append(f"{cid}: path {r['path']} does not match its split and label")
            if not os.path.exists(self.abs_path(r)):
                problems.append(f"{cid}: file missing: {r['path']}")
            group_splits.setdefault(r["group_id"], set()).add(r["split"])
            if r["origin"] == "augmented":
                parent = self.by_id.get(r["parent_crop_id"])
                if r["split"] != "train":
                    problems.append(f"{cid}: augmented copy in {r['split']}")
                if parent is None or parent["group_id"] != r["group_id"]:
                    problems.append(f"{cid}: parent {r['parent_crop_id']} missing or in another group")
        for g, splits in group_splits.items():
            if len(splits) > 1:
                problems.append(f"group {g} spans splits {sorted(splits)}")
        return problems


def seed_from_dataset_bin(pool, dataset_bin, added_in="seed"):
    """Copy dataset_bin (split_bin_dataset.py output) into the pool:
    train -> train, val -> val, test -> test_fixed; group_id = item_id."""
    with open(os.path.join(dataset_bin, "manifest.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    added = Counter()
    for r in rows:
        if r["crop_id"] in pool.by_id:
            continue                                     # seeding again is harmless
        split = DATASET_BIN_SPLITS[r["split"]]
        pool.add(os.path.join(dataset_bin, *r["new_path"].split("/")), r["crop_id"], split, r["label"],
                 r["item_id"], "capture", added_in=added_in)
        added[split] += 1
    return added


def seed_from_organik(pool, dataset_dir, added_in="seed"):
    """Copy dataset_organik_v2 (release data-organik-v2, Sian's similarity-grouped split) into the pool:
    train -> train, val -> val, test -> test_fixed; origin public; group_id = organik_g<group>, so a
    group that spans two splits is refused (pool.add). crop_id = organik_<file stem>."""
    with open(os.path.join(dataset_dir, "manifest.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = [f"organik_{os.path.splitext(r['filename'])[0]}" for r in rows]
    dupes = [c for c, n in Counter(ids).items() if n > 1]
    if dupes:
        raise PoolError(f"{len(dupes)} file names repeat in {dataset_dir}, e.g. {dupes[:3]}")
    added = Counter()
    for r, crop_id in zip(rows, ids):
        if crop_id in pool.by_id:
            continue                                     # seeding again is harmless
        split = DATASET_BIN_SPLITS[r["split"]]
        pool.add(os.path.join(dataset_dir, *r["new_path"].replace("\\", "/").split("/")), crop_id, split,
                 FOLDER_LABELS[r["class_folder"]], f"organik_g{r['group']}", "public", added_in=added_in)
        added[split] += 1
    return added


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create, seed and check the shared pool")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed", help="copy a split dataset into the pool (train, val, test_fixed)")
    s.add_argument("--pool", required=True)
    src = s.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset_bin", help="split_bin_dataset.py output (bin-camera captures)")
    src.add_argument("--organik", help="dataset_organik_v2 (release data-organik-v2)")
    c = sub.add_parser("check", help="list rule violations and counts")
    c.add_argument("--pool", required=True)
    a = ap.parse_args()

    pool = Pool(a.pool)
    if a.cmd == "seed":
        added = seed_from_organik(pool, a.organik) if a.organik else seed_from_dataset_bin(pool, a.dataset_bin)
        pool.save()
        print(f"seeded {sum(added.values())} images: {dict(added)}")
    problems = pool.check()
    for (split, label), n in sorted(pool.counts().items()):
        print(f"  {split:11s} {label}: {n}")
    print("\n".join(problems) if problems else "pool OK: no rule violations")
    raise SystemExit(1 if problems else 0)
