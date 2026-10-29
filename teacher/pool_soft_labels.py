"""Soft labels for the shared pool: the teacher's side of update/run_update.py (step 2).

  python teacher/pool_soft_labels.py --pool pool --version teacher_v2 --out runs/teacher_v2/soft_labels.csv

The same teacher as teacher.py (frozen DINOv2 ViT-B/14 features + logistic regression, C and the
temperature picked on val), fitted on the pool's train rows: the Organik images, the feedback crops and
the augmented copies. Train rows get out-of-fold soft labels: 5 folds by group_id, so an item, its load
and its augmented copies stay on one side. The pool holds the detector's crops already, so nothing is
cropped.

Writes one row per train crop_id (run_update.py checks for a row for every train image and none for
val or test): crop_id, teacher_logit, teacher_wet_prob, fold, teacher_version. The student reads it
with common.SoftLabels (pool layout). Also saves <out dir>/<version>.joblib and prints the teacher's
accuracy on val and test_fixed (for the record: nothing is tuned on test_fixed).

Pool-only: the public datasets (Roboflow, RealWaste) are not added here.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teacher.teacher import CLASSES, FeatureExtractor, Teacher, build_training_views, out_of_fold_logits, sigmoid  # noqa: E402


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


def fold_of_groups(X, y, groups, n_folds=5):
    """{group: fold} as out_of_fold_logits() splits them (same GroupKFold, same order)."""
    from sklearn.model_selection import GroupKFold
    k = min(n_folds, len(np.unique(groups)))
    out = {}
    for fold, (tr, _) in enumerate(GroupKFold(n_splits=k).split(X, y, groups)):
        for g in np.setdiff1d(np.unique(groups), np.unique(groups[tr])):
            out[g] = fold
    return out


def main(a):
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pool_frame(a.pool)
    train = df[df.split == "train"].reset_index(drop=True)
    val = df[df.split == "val"].reset_index(drop=True)
    if train.empty or val.empty:
        raise SystemExit("the pool needs train and val images")
    ext = FeatureExtractor(cache_dir=out.parent / "cache")
    X, y, g, v = build_training_views(ext, train, a.n_aug)
    Xv = ext.features(val)
    t = Teacher(a.version).fit(X, y, Xv, val.y.values, trained_on={
        "pool": str(a.pool), "n_train": len(train), "n_val": len(val),
        "origins": train.source.value_counts().to_dict()})
    print(f"{a.version}: C={t.C}, T={t.temperature:.3f}; trained on {len(train)} train images "
          f"{train.source.value_counts().to_dict()}")
    z = out_of_fold_logits(X, y, g, v, t.C)
    folds = fold_of_groups(X, y, g)
    table = pd.DataFrame({
        "crop_id": train.crop_id, "teacher_logit": z, "teacher_wet_prob": sigmoid(z),
        "fold": [folds[x] for x in train.group], "teacher_version": a.version})
    if table.crop_id.duplicated().any():
        raise SystemExit("duplicate crop_ids in the pool's train rows")
    table.to_csv(out, index=False)
    t.save(out.parent / f"{a.version}.joblib")
    oof_acc = float(((table.teacher_wet_prob >= 0.5).astype(int) == train.y.values).mean())
    pv = t.wet_prob(Xv)
    print(f"out-of-fold accuracy on train {oof_acc:.4f}; val accuracy {float(((pv >= 0.5) == val.y.values).mean()):.4f}")
    test = df[df.split == "test_fixed"].reset_index(drop=True)
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
    ap.add_argument("--n_aug", type=int, default=2, help="augmented views per train image for fitting")
    main(ap.parse_args())