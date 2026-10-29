"""Soft labels for the shared pool: the teacher's side of update/run_update.py (step 2).

  python teacher/pool_soft_labels.py --pool pool --version teacher_v2 \
      --gray dataset_D_v2_balanced --out runs/teacher_v2/soft_labels.csv

The teacher must NOT learn from the student's data (Organik). So:

  Fitted on   : the public datasets (--gray, optional --plate, cropped with the D masks, as in
                teacher.py / teacher_v1_gray) + the pool's human-checked crops (feedback) and
                their augmented copies.
  Tuned on    : public val + the pool's non-Organik val rows (C and temperature).
  Never seen  : Organik (any split), anything sharing a group_id with an Organik row (its
                augmented copies), replayed-Organik feedback (replay-* groups), and test_fixed.

How Organik is found: by group_id prefix (default organik_g, so organik_g<N>). In the real pool the
Organik rows have origin = public and their augmented copies origin = augmented, but both carry the
organik_g<N> group, so the prefix catches both. --exclude_origins can add whole origins on top.

Labels still go to EVERY pool train row, Organik included, because run_update.py expects a row
for every train image. Labelling Organik is fine; only fitting on it is not.
  - Organik rows   : the full teacher's raw logit (it never trained on them).
                     fold = a group-based fold tag (5 folds by group_id), kept for the format only.
  - Feedback rows  : out-of-fold logit (5 folds by group, over the whole fit set), so a crop never
                     gets a label from a model that trained on it.

Leak guard: stops with an error if any image the teacher is fitted on has the same file bytes
(sha256) as an Organik image or a test_fixed image in the pool. It cannot catch copies of Organik
photos that are not in the pool (e.g. held-back photos replayed as feedback), which is why replay-*
groups are excluded by default.

Writes one row per train crop_id: crop_id, teacher_logit, teacher_wet_prob, fold, teacher_version.
Also saves <out dir>/<version>.joblib and prints accuracy on val and test_fixed (for the record:
nothing is tuned on test_fixed).

Before running with real crops: do not reuse a pool from a test run of the loop. Its "feedback" crops
were replayed Organik photos.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teacher.teacher import (CLASSES, FeatureExtractor, Teacher, build_training_views,  # noqa: E402
                             find_masks_dir, index_split_dataset, out_of_fold_logits, sigmoid)


def pool_frame(pool_dir):
    """The pool's manifest as the frame teacher.py works on (one row per image, in manifest order)."""
    pool_dir = Path(pool_dir)
    m = pd.read_csv(pool_dir / "manifest.csv", dtype=str, keep_default_na=False)
    bad = set(m["label"]) - {"dry", "wet"}
    if bad:
        raise SystemExit(f"unknown labels in manifest: {bad}")
    y = (m["label"] == "wet").astype(int)
    return pd.DataFrame({
        "dataset": "pool", "split": m["split"], "class_folder": [CLASSES[i] for i in y],
        "filename": [os.path.basename(p) for p in m["path"]],
        "path": [str(pool_dir.joinpath(*p.split("/"))) for p in m["path"]],
        "y": y, "mask_path": None, "crop": False, "source": m["origin"], "group": m["group_id"],
        "crop_id": m["crop_id"]})


def group_folds(groups, n_folds=5):
    """Fold tag per row, by group (GroupKFold), for rows that get no out-of-fold label."""
    from sklearn.model_selection import GroupKFold
    groups = np.asarray(groups)
    k = min(n_folds, len(np.unique(groups)))
    if k < 2:
        return np.zeros(len(groups), dtype=int)
    out = np.full(len(groups), -1, dtype=int)
    for fold, (_, held) in enumerate(GroupKFold(n_splits=k).split(groups, groups=groups)):
        out[held] = fold
    return out


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def leak_guard(fit_df, protected_df):
    """Stop if any fitted image is byte-identical to a protected (Organik / test_fixed) image."""
    protected = {file_sha(p): p for p in protected_df.path}
    hits = [(p, protected[h]) for p in fit_df.path if (h := file_sha(p)) in protected]
    if hits:
        lines = "\n".join(f"  {a}  ==  {b}" for a, b in hits[:20])
        raise SystemExit(f"LEAK: {len(hits)} fitted image(s) are copies of Organik/test_fixed images:\n{lines}")
    print(f"leak guard: 0 of {len(fit_df)} fitted images match any of {len(protected_df)} Organik/test_fixed images")


def public_frames(a):
    masks = a.masks or (find_masks_dir(a.gray) if a.gray else None)
    frames = []
    if a.gray:
        frames.append(index_split_dataset(a.gray, "dataset_D_v2_balanced", masks, True))
    if a.plate:
        if masks is None:
            print("warning: no D masks found; C images will be used uncropped")
        frames.append(index_split_dataset(a.plate, "dataset_C_v2", masks, True))
    return pd.concat(frames, ignore_index=True)


def main(a):
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- pool: split Organik (label only) from feedback (fit + label) ---------------------
    pool = pool_frame(a.pool)
    print("pool images by origin and split:")
    print(pool.groupby(["source", "split"]).size().to_string())
    prefixes = tuple(x.strip() for x in a.organik_groups.split(",") if x.strip())
    excl = {o.strip().lower() for o in a.exclude_origins.split(",") if o.strip()}
    is_org = pool.group.str.startswith(prefixes) | pool.source.str.lower().isin(excl)
    if not is_org.any():
        raise SystemExit(f"no Organik rows found (group prefix {prefixes}, origins {sorted(excl)}); "
                         "check --organik_groups against the pool's group_id values")
    is_replay = pool.group.str.startswith(tuple(x.strip() for x in a.replay_groups.split(",") if x.strip()))
    if is_replay.any():
        print(f"leaving out {int(is_replay.sum())} replayed-Organik rows ({a.replay_groups} groups)")
    org_groups = set(pool.group[is_org])
    blocked = is_org | is_replay | pool.group.isin(org_groups)   # Organik + its augmented copies + replays
    print(f"Organik rows: {int(is_org.sum())} (by origin: "
          f"{pool.source[is_org].value_counts().to_dict()})")

    pool_train = pool[pool.split == "train"].reset_index(drop=True)
    tr_blocked = blocked[pool.split == "train"].values
    fit_pool = pool_train[~tr_blocked]
    label_only = pool_train[tr_blocked]
    val_pool = pool[(pool.split == "val") & ~blocked]
    print(f"pool train: {len(fit_pool)} feedback rows (fit + label), {len(label_only)} Organik rows (label only)")

    # --- fit set: public + feedback ---------------------------------------------------------
    pub = public_frames(a)
    pub_train = pub[pub.split == "train"]
    pub_val = pub[pub.split == "val"]
    fit_df = pd.concat([pub_train, fit_pool.assign(dataset="pool_fit")], ignore_index=True)
    val_df = pd.concat([pub_val, val_pool.assign(dataset="pool_fit")], ignore_index=True)
    if fit_df.empty or val_df.empty:
        raise SystemExit("need train and val images from the public datasets")
    leak_guard(fit_df, pool[is_org | (pool.split == "test_fixed")])

    ext = FeatureExtractor(cache_dir=out.parent / "cache")
    X, y, g, v = build_training_views(ext, fit_df, a.n_aug)
    Xv = ext.features(val_df)
    t = Teacher(a.version).fit(X, y, Xv, val_df.y.values, trained_on={
        "public": sorted(pub_train.dataset.unique()), "n_public_train": len(pub_train),
        "pool": str(a.pool), "n_feedback_train": len(fit_pool), "n_val": len(val_df),
        "organik_group_prefixes": list(prefixes), "excluded_origins": sorted(excl), "pool_rows_not_fitted": int(blocked.sum())})
    print(f"{a.version}: C={t.C}, T={t.temperature:.3f}; fitted on {len(pub_train)} public + "
          f"{len(fit_pool)} feedback train images (no Organik)")

    # --- soft labels for every pool train row -----------------------------------------------
    z = np.full(len(pool_train), np.nan)
    fold = np.full(len(pool_train), -1, dtype=int)
    fit_idx = np.where(~tr_blocked)[0]
    lab_idx = np.where(tr_blocked)[0]
    if len(fit_idx):
        z_oof, f_oof = out_of_fold_logits(X, y, g, v, t.C, return_folds=True)
        z[fit_idx] = z_oof[len(pub_train):]                # view-0 rows come in fit_df order
        fold[fit_idx] = f_oof[len(pub_train):]
    if len(lab_idx):
        z[lab_idx] = t.logit(ext.features(label_only.reset_index(drop=True)))
        fold[lab_idx] = group_folds(label_only.group.values)
    assert not np.isnan(z).any() and (fold >= 0).all(), "some pool train rows got no label"

    table = pd.DataFrame({
        "crop_id": pool_train.crop_id, "teacher_logit": z, "teacher_wet_prob": sigmoid(z),
        "fold": fold, "teacher_version": a.version})
    if table.crop_id.duplicated().any():
        raise SystemExit("duplicate crop_ids in the pool's train rows")
    table.to_csv(out, index=False)
    t.save(out.parent / f"{a.version}.joblib")

    pred = (table.teacher_wet_prob >= 0.5).astype(int).values
    if len(lab_idx):
        print(f"accuracy on Organik train (never fitted) {float((pred[lab_idx] == pool_train.y.values[lab_idx]).mean()):.4f}")
    if len(fit_idx):
        print(f"out-of-fold accuracy on feedback train {float((pred[fit_idx] == pool_train.y.values[fit_idx]).mean()):.4f}")
    pv = t.wet_prob(Xv)
    print(f"val accuracy {float(((pv >= 0.5) == val_df.y.values).mean()):.4f} ({len(val_df)} images)")
    test = pool[pool.split == "test_fixed"].reset_index(drop=True)
    if len(test):
        pt = t.wet_prob(ext.features(test))
        print(f"(for the record) test_fixed accuracy {float(((pt >= 0.5) == test.y.values).mean()):.4f}, "
              f"{len(test)} images; not used for any choice")
    print(f"wrote {len(table)} soft labels -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Soft labels for the shared pool (teacher side of run_update.py)")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--version", required=True, help="teacher_v(N+1) for update N, e.g. teacher_v2")
    ap.add_argument("--out", required=True, help="<runs>/teacher_vN/soft_labels.csv")
    ap.add_argument("--gray", required=True, help="dataset_D_v2_balanced folder (public data, as teacher_v1_gray)")
    ap.add_argument("--plate", help="dataset_C_v2 folder (optional, public data)")
    ap.add_argument("--masks", help="masks folder (default: found inside --gray)")
    ap.add_argument("--organik_groups", default="organik_g",
                    help="comma-separated group_id prefixes of Organik rows (catches originals and augmented copies)")
    ap.add_argument("--replay_groups", default="replay-",
                    help="comma-separated group_id prefixes of replayed-Organik feedback, always left out")
    ap.add_argument("--exclude_origins", default="",
                    help="optional: whole pool origins to treat as Organik too (case-insensitive), e.g. public")
    ap.add_argument("--n_aug", type=int, default=2, help="augmented views per train image for fitting")
    main(ap.parse_args())