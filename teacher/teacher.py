"""Teacher model for the AI dustbin: frozen DINOv2 ViT-B/14 features + logistic regression.

Runs off the bin (Colab / laptop). In the update cycle it:
  1. indexes training data: C/D (cropped to the item), bin dataset, human feedback
  2. extracts DINOv2 features (cached)
  3. fits the classifier (C chosen on val) and a temperature (on val)
  4. exports soft labels for the student (format 5, out-of-fold for train)
  5. saves itself (format 6)

Hand-off formats (team doc):
  2  bin dataset : dataset_bin/{train,val,test}/{dry waste,wet waste}/<crop_id>.jpg + manifest.csv
                   (crop_id, new_path, split, label, item_id, session, lighting, captured_at)
  4  feedback    : hitl.py FeedbackStore folder: feedback.csv (crop_file relative to the folder,
                   e.g. crops/<crop_id>.jpg; load_id, box, human_label, ...). Read through
                   student/common.py load_feedback(), so teacher and student use the same rows.
  5  soft labels : split, class_folder, filename, teacher_wet_prob, teacher_logit, teacher_version
                   teacher_logit = raw log-odds of "wet" (uncalibrated); teacher_wet_prob = sigmoid(logit / T)
  6  teacher     : teacher_vN.joblib = classifier, temperature, backbone, crop settings, ...

Torch is imported lazily, so indexing / fitting / soft-label code also runs without a GPU.
"""
from __future__ import annotations

import datetime as _dt
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Shared constants: taken from student/common.py so teacher and student crop
# and preprocess identically. Fallback copy only if student/ isn't importable.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
try:
    from student.common import (CLASSES, EVAL_PAD, GRAY, IMG_EXTS, IMG_SIZE, LABELS, MEAN,
                                STD, TRAIN_PAD, crop_to_item)
    COMMON_SOURCE = "student/common.py"
except ImportError:  # keep in sync with student/common.py
    GRAY, IMG_SIZE = 128, 224
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    CLASSES, LABELS = ["dry waste", "wet waste"], ["dry", "wet"]
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    EVAL_PAD, TRAIN_PAD = (0.10, 0.10), (0.05, 0.30)
    MIN_MASK_FRAC = 0.01

    def crop_to_item(im, mask, pad):
        if mask is None:
            return im
        m = np.asarray(mask.convert("L")) > 127
        if m.mean() < MIN_MASK_FRAC:
            return im
        ys, xs = np.nonzero(m)
        sx, sy = im.width / m.shape[1], im.height / m.shape[0]
        x0, x1, y0, y1 = xs.min() * sx, (xs.max() + 1) * sx, ys.min() * sy, (ys.max() + 1) * sy
        p = max(x1 - x0, y1 - y0) * pad
        box = (max(0, int(x0 - p)), max(0, int(y0 - p)), min(im.width, int(x1 + p)), min(im.height, int(y1 + p)))
        return im.crop(box)
    COMMON_SOURCE = "fallback copy in teacher.py"

SPLITS = ["train", "val", "test"]
BACKBONE = "dinov2_vitb14"
CROP_SETTINGS = {"eval_pad": list(EVAL_PAD), "train_pad": list(TRAIN_PAD), "img_size": IMG_SIZE,
                 "gray": GRAY, "masks": "dataset_D_v2_balanced/masks/{split}/{class}/{stem}.png",
                 "resize": "squash to img_size x img_size", "normalise": "ImageNet"}
SOFT_LABEL_COLS = ["split", "class_folder", "filename", "teacher_wet_prob", "teacher_logit", "teacher_version"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


# ===========================================================================
# 1. Indexing
# ===========================================================================
def find_dataset_root(path) -> Path:
    """Folder that directly contains train/ val/ test/ (zips are often nested one level)."""
    path = Path(path)
    for p in [path, *sorted(q for q in path.rglob("*") if q.is_dir())]:
        if all((p / s).is_dir() for s in SPLITS):
            return p
    raise FileNotFoundError(f"no folder with train/ val/ test/ under {path}")


def find_masks_dir(path):
    """masks/ folder of dataset D (masks/{split}/{class}/{stem}.png), or None."""
    path = Path(path)
    for p in [path / "masks", *path.rglob("masks")]:
        if p.is_dir() and any((p / s).is_dir() for s in SPLITS):
            return p
    return None


def _path_key(p):
    """(split, class_folder, stem) from any path ending in .../<split>/<class>/<file>."""
    parts = str(p).replace("\\", "/").split("/")
    return parts[-3], parts[-2], os.path.splitext(parts[-1])[0]


def index_split_dataset(root, dataset, masks_dir=None, crop=True, source=None) -> pd.DataFrame:
    """One row per image of <root>/{split}/{class}/<file>.

    Columns: dataset, split, class_folder, filename, path, y, source, group, mask_path, crop.
    Matching with the manifest is on split + class folder + file stem (file names repeat
    across folders, e.g. two different Capture2.JPG). The manifest's new_path is used,
    with \\ turned into /.
    """
    root = find_dataset_root(root)
    rows = []
    for split in SPLITS:
        for y, cls in enumerate(CLASSES):
            folder = root / split / cls
            if not folder.is_dir():
                raise FileNotFoundError(f"missing folder: {folder}")
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    mp = (Path(masks_dir) / split / cls / (p.stem + ".png")) if masks_dir else None
                    rows.append(dict(dataset=dataset, split=split, class_folder=cls, filename=p.name,
                                     path=str(p), y=y, mask_path=str(mp) if mp and mp.exists() else None,
                                     crop=bool(crop and masks_dir)))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"no images under {root}")
    df["source"] = source or "unknown"
    stems = df.filename.map(lambda f: os.path.splitext(f)[0])
    df_keys = (df.split + "|" + df.class_folder + "|" + stems).tolist()
    # default out-of-fold group = the photo itself; C and D share it because they hold the same photos
    df["group"] = ["img/" + k for k in df_keys]

    man_path = root / "manifest.csv"
    if man_path.exists():
        man = pd.read_csv(man_path)
        if "new_path" not in man.columns:
            raise KeyError(f"{man_path} has no new_path column: {list(man.columns)}")
        man["_key"] = ["|".join(_path_key(p)) for p in man.new_path.astype(str)]
        if man._key.duplicated().any():
            raise ValueError(f"{man_path}: {man._key.duplicated().sum()} duplicate split/class/file keys")
        man = man.set_index("_key")
        hit = [k in man.index for k in df_keys]
        print(f"{dataset}: matched {sum(hit)}/{len(df)} images to manifest.csv")
        if "source" in man.columns and source is None:
            src = man["source"].to_dict()
            df["source"] = [src[k] if h else "unknown" for k, h in zip(df_keys, hit)]
        # keep near-duplicates / photos of one bin item together in out-of-fold splits
        gcol = next((c for c in man.columns if c == "item_id" or "group" in c.lower()), None)
        if gcol:
            gm = man[gcol].to_dict()
            df["group"] = [f"{gcol}/{gm[k]}" if h and pd.notna(gm[k]) else old
                           for k, h, old in zip(df_keys, hit, df.group)]
    else:
        print(f"{dataset}: no manifest.csv, source = {df.source.iloc[0]}")
    if masks_dir and crop:
        print(f"{dataset}: {df.mask_path.notna().sum()}/{len(df)} images have a mask (others used uncropped)")
    return df.reset_index(drop=True)


def index_bin_dataset(root) -> pd.DataFrame:
    """Format 2. Bin crops are already the detector's crops: not cropped again."""
    df = index_split_dataset(root, "bin", masks_dir=None, crop=False, source="bin")
    return df


def index_feedback(feedback_dir, exclude_hashes=frozenset()) -> pd.DataFrame:
    """Format 4 (hitl.py FeedbackStore folder) → training rows (split='feedback').

    Which crops are used is decided by student/common.py load_feedback(), so the teacher and
    the student always learn from exactly the same feedback:
      * every item the human labelled is kept, including items of dry+wet loads;
      * dropped: no human label, missing file, a crop whose box overlaps another item's box
        in the same load, and exact copies of val/test images (exclude_hashes).
    Crop paths come from feedback.csv's crop_file (relative to the folder, e.g. crops/<id>.jpg).
    Rows are grouped by load_id for the out-of-fold soft labels.
    """
    try:
        from student.common import load_feedback
    except ImportError as e:
        raise ImportError("index_feedback needs student/common.py (load_feedback), so teacher and "
                          "student use the same feedback rule") from e
    feedback_dir = Path(feedback_dir)
    used = load_feedback(feedback_dir, exclude_hashes)
    fb = pd.read_csv(feedback_dir / "feedback.csv", dtype=str, keep_default_na=False)
    by_path = {str((feedback_dir / r["crop_file"]).resolve()): r for r in fb.to_dict("records")}
    rows = []
    for path, y in used:
        r = by_path.get(str(Path(path).resolve()), {})
        rows.append(dict(dataset="feedback", split="feedback", class_folder=CLASSES[y], filename=Path(path).name,
                         path=str(path), y=y, mask_path=None, crop=False, source="feedback",
                         group=f"feedback/load/{r.get('load_id') or Path(path).stem}",
                         trigger=r.get("trigger"), student_version=r.get("student_version")))
    return pd.DataFrame(rows, columns=["dataset", "split", "class_folder", "filename", "path", "y", "mask_path",
                                       "crop", "source", "group", "trigger", "student_version"])


def val_test_hashes(roots):
    """sha256 of every val/test image of the given dataset folders (feedback leak guard)."""
    from student.common import file_hashes
    out = set()
    for root in roots:
        out |= file_hashes(find_dataset_root(root))
    return out


# ===========================================================================
# 2. Images and features
# ===========================================================================
def load_item_image(row, pad=EVAL_PAD, rng=None) -> Image.Image:
    """RGB PIL image framed like the detector's crop (for C/D: crop_to_item with the D mask)."""
    with Image.open(row["path"]) as im:
        im = im.convert("RGB")
    if row.get("crop") and row.get("mask_path"):
        lo, hi = pad
        p = lo if lo == hi else float((rng or np.random.default_rng()).uniform(lo, hi))
        with Image.open(row["mask_path"]) as mask:
            im = crop_to_item(im, mask, p)
    return im


class FeatureExtractor:
    """Frozen DINOv2 CLS features (768-d). One instance holds the backbone on the GPU."""

    def __init__(self, backbone=BACKBONE, device=None, cache_dir=None):
        import torch
        from torchvision import transforms as T
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone_name = backbone
        self.model = torch.hub.load("facebookresearch/dinov2", backbone).to(self.device).eval()
        self.eval_tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])
        self.train_tf = T.Compose([T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
                                   T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2),
                                   T.ToTensor(), T.Normalize(MEAN, STD)])
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def embed_pil(self, images, tf=None, batch_size=64):
        torch = self.torch
        tf = tf or self.eval_tf
        out = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                x = torch.stack([tf(im) for im in images[i:i + batch_size]]).to(self.device)
                with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                    out.append(self.model(x).float().cpu())
        return torch.cat(out).numpy() if out else np.zeros((0, 768), np.float32)

    def features(self, df, view=0, seed=0, batch_size=64):
        """Features for every row of df, in order.

        view 0 = the detector's framing (EVAL_PAD, no augmentation). view k>0 = a training
        view: random pad from TRAIN_PAD and Sian's training augmentation, seeded by k.
        Cached per (dataset, split, view) in cache_dir, checked against the row keys.
        """
        if len(df) == 0:
            return np.zeros((0, 768), np.float32)
        parts = []
        for (ds, sp), chunk in df.groupby(["dataset", "split"], sort=False):
            parts.append((chunk.index, self._features_chunk(chunk, view, seed, f"{ds}_{sp}", batch_size)))
        X = np.zeros((len(df), parts[0][1].shape[1]), np.float32)
        pos = pd.Series(np.arange(len(df)), index=df.index)
        for idx, Xc in parts:
            X[pos.loc[idx].values] = Xc
        return X

    def _features_chunk(self, df, view, seed, tag, batch_size):
        keys = (df.dataset + "|" + df.split + "|" + df.class_folder + "|" + df.filename).values
        cpath = self.cache_dir / f"feats_{tag}_v{view}.npz" if self.cache_dir else None
        if cpath and cpath.exists():
            d = np.load(cpath, allow_pickle=True)
            if len(d["keys"]) == len(keys) and (d["keys"] == keys).all():
                return d["X"]
        rng = np.random.default_rng(seed * 1000 + view)
        torch = self.torch
        torch.manual_seed(seed * 1000 + view)
        pad, tf = (EVAL_PAD, self.eval_tf) if view == 0 else (TRAIN_PAD, self.train_tf)
        X = []
        rows = df.to_dict("records")
        for i in range(0, len(rows), batch_size):
            ims = [load_item_image(r, pad, rng) for r in rows[i:i + batch_size]]
            X.append(self.embed_pil(ims, tf, batch_size))
        X = np.concatenate(X) if X else np.zeros((0, 768), np.float32)
        if cpath:
            np.savez(cpath, X=X, keys=keys)
        return X


def build_training_views(extractor, train_df, n_aug=2):
    """Stack view 0 + n_aug augmented views. Returns X, y, groups, view_ids."""
    Xs = [extractor.features(train_df, view=v) for v in range(n_aug + 1)]
    n = len(train_df)
    X = np.concatenate(Xs)
    y = np.tile(train_df.y.values, n_aug + 1)
    groups = np.tile(train_df.group.values, n_aug + 1)
    views = np.repeat(np.arange(n_aug + 1), n)
    return X, y, groups, views


# ===========================================================================
# 3. The teacher
# ===========================================================================
def _make_clf(C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def fit_temperature(logits, y):
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import log_loss
    nll = lambda lt: log_loss(y, np.clip(sigmoid(logits / np.exp(lt)), 1e-7, 1 - 1e-7), labels=[0, 1])
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


class Teacher:
    """Linear classifier + temperature on top of frozen DINOv2 features."""

    C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)

    def __init__(self, version="teacher_v1"):
        self.version = version
        self.clf, self.C, self.temperature = None, None, 1.0
        self.backbone = BACKBONE
        self.crop_settings = dict(CROP_SETTINGS)
        self.trained_on = {}
        self.extractor = None    # set lazily for classify_image()

    # -- fitting ---------------------------------------------------------
    def fit(self, X, y, X_val, y_val, C=None, trained_on=None):
        """Pick C on val log-loss (unless given), refit, then fit the temperature on val."""
        from sklearn.metrics import log_loss
        if C is None:
            best = None
            for c in self.C_GRID:
                clf = _make_clf(c).fit(X, y)
                l = log_loss(y_val, clf.predict_proba(X_val)[:, 1], labels=[0, 1])
                if best is None or l < best[0]:
                    best = (l, c, clf)
            _, self.C, self.clf = best
        else:
            self.C, self.clf = C, _make_clf(C).fit(X, y)
        self.temperature = fit_temperature(self.logit(X_val), y_val)
        self.trained_on = trained_on or {}
        return self

    def refit(self, X_old, y_old, X_new, y_new, X_val, y_val, keep_C=True, trained_on=None):
        """Feedback round: refit on old data + new labels (not new only, to avoid forgetting)."""
        return self.fit(np.concatenate([X_old, X_new]), np.concatenate([y_old, y_new]),
                        X_val, y_val, C=self.C if keep_C else None, trained_on=trained_on)

    # -- prediction ------------------------------------------------------
    def logit(self, X):
        """Raw log-odds of wet (uncalibrated)."""
        return self.clf.decision_function(X)

    def wet_prob(self, X):
        """Calibrated P(wet)."""
        return sigmoid(self.logit(X) / self.temperature)

    def classify_image(self, crop_bgr):
        """Same interface as the student: one BGR uint8 crop (the detector's crop) in,
        {"label": "dry"/"wet", "confidence", "dry_prob", "wet_prob", "inference_ms"} out."""
        t0 = time.time()
        if self.extractor is None:
            self.extractor = FeatureExtractor(self.backbone)
        im = Image.fromarray(np.ascontiguousarray(crop_bgr[:, :, ::-1]))
        p = float(self.wet_prob(self.extractor.embed_pil([im]))[0])
        return {"label": LABELS[int(p >= 0.5)], "confidence": max(p, 1 - p), "dry_prob": 1 - p,
                "wet_prob": p, "inference_ms": (time.time() - t0) * 1000}

    # -- format 6 --------------------------------------------------------
    def save(self, path):
        import joblib
        import sklearn
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"classifier": self.clf, "C": self.C, "temperature": self.temperature,
                     "backbone": self.backbone, "crop_settings": self.crop_settings,
                     "teacher_version": self.version, "trained_on": self.trained_on,
                     "labels": LABELS, "classes": CLASSES, "sklearn_version": sklearn.__version__,
                     "created": _dt.datetime.now().isoformat(timespec="seconds")}, path)
        return path

    @classmethod
    def load(cls, path):
        import joblib
        d = joblib.load(path)
        t = cls(d["teacher_version"])
        t.clf, t.C, t.temperature = d["classifier"], d.get("C"), d["temperature"]
        t.backbone, t.crop_settings, t.trained_on = d["backbone"], d["crop_settings"], d.get("trained_on", {})
        return t


# ===========================================================================
# 4. Soft labels (format 5)
# ===========================================================================
def out_of_fold_logits(X, y, groups, views, C, n_folds=5, seed=0):
    """Logit for every view-0 row from a model that never saw that row's group.

    X/y/groups/views come from build_training_views (all views stacked). Folds are by group,
    so augmented views, near-duplicates and photos of one bin item stay on one side.
    """
    from sklearn.model_selection import GroupKFold
    clean = np.where(views == 0)[0]
    z = np.full(len(clean), np.nan)
    n_groups = len(np.unique(groups))
    k = min(n_folds, n_groups)
    gkf = GroupKFold(n_splits=k)
    for tr, _ in gkf.split(X, y, groups):
        held_groups = np.setdiff1d(np.unique(groups), np.unique(groups[tr]))
        held = np.isin(groups[clean], held_groups)
        z[held] = _make_clf(C).fit(X[tr], y[tr]).decision_function(X[clean[held]])
    assert not np.isnan(z).any(), "some rows got no out-of-fold prediction"
    return z


def soft_label_table(df, logits, temperature, version):
    """Format 5 rows for df (any split), given raw logits in df order."""
    out = df[["split", "class_folder", "filename"]].copy()
    out["teacher_wet_prob"] = sigmoid(np.asarray(logits) / temperature)
    out["teacher_logit"] = np.asarray(logits, dtype=float)
    out["teacher_version"] = version
    return out[SOFT_LABEL_COLS]


def write_soft_labels(table, path):
    key = table.split + "|" + table.class_folder + "|" + table.filename
    if key.duplicated().any():
        raise ValueError(f"{key.duplicated().sum()} duplicate split/class/filename keys in soft labels")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


# ===========================================================================
# 5. Evaluation
# ===========================================================================
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def ece(p_wet, y, bins=15):
    p_wet, y = np.asarray(p_wet), np.asarray(y)
    conf = np.maximum(p_wet, 1 - p_wet)
    pred = (p_wet >= 0.5).astype(int)
    e, edges = 0.0, np.linspace(0.5, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ((conf > lo) | (lo == 0.5)) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def evaluate(name, p_wet, df):
    """Accuracy with 95% Wilson CI, overall and per source; ECE; confusion matrix."""
    y = df.y.values
    pred = (np.asarray(p_wet) >= 0.5).astype(int)
    ok = pred == y
    lo, hi = wilson_ci(ok.sum(), len(ok))
    r = {"model": name, "n": len(ok), "acc": ok.mean(), "acc_ci95": f"[{lo:.3f}, {hi:.3f}]",
         "ece": ece(p_wet, y),
         "cm [[dd,dw],[wd,ww]]": [[int(((y == a) & (pred == b)).sum()) for b in (0, 1)] for a in (0, 1)]}
    for s in sorted(df.source.unique()):
        m = (df.source == s).values
        lo, hi = wilson_ci(ok[m].sum(), m.sum())
        r[f"acc_{s}"] = f"{ok[m].mean():.3f} [{lo:.3f}, {hi:.3f}] n={m.sum()}"
    return r


# ===========================================================================
# 6. One update cycle (CLI): fit on everything available, export soft labels, save
# ===========================================================================
def run_update(datasets, out_dir, version, feedback_dir=None, n_aug=2, cache_dir=None, extractor=None):
    """datasets: {name: (root, masks_dir or None, crop: bool)}. Train = all train splits +
    feedback; val = all val splits (C and the temperature are chosen there). Writes
    soft_labels_<dataset>_<version>.csv per dataset (+ feedback) and <version>.joblib."""
    out_dir = Path(out_dir)
    ext = extractor or FeatureExtractor(cache_dir=cache_dir or out_dir / "cache")
    frames = [index_split_dataset(root, name, masks, crop) if name != "bin" else index_bin_dataset(root)
              for name, (root, masks, crop) in datasets.items()]
    if feedback_dir:
        fb = index_feedback(feedback_dir, val_test_hashes([root for root, _, _ in datasets.values()]))
        if len(fb):
            frames.append(fb)
    all_df = pd.concat(frames, ignore_index=True)
    train = all_df[all_df.split.isin(["train", "feedback"])].reset_index(drop=True)
    val = all_df[all_df.split == "val"].reset_index(drop=True)
    test = all_df[all_df.split == "test"].reset_index(drop=True)

    X, y, g, v = build_training_views(ext, train, n_aug)
    Xv = ext.features(val)
    Xt = ext.features(test)
    t = Teacher(version).fit(X, y, Xv, val.y.values, trained_on={
        "datasets": list(datasets), "feedback_rows": int((train.split == "feedback").sum()),
        "n_train": len(train), "n_val": len(val)})
    print(f"{version}: C={t.C}, T={t.temperature:.3f}")

    z_oof = out_of_fold_logits(X, y, g, v, t.C)
    tables = pd.concat([train.assign(_z=z_oof), val.assign(_z=t.logit(Xv)), test.assign(_z=t.logit(Xt))])
    for name, part in tables.groupby("dataset"):
        write_soft_labels(soft_label_table(part, part._z.values, t.temperature, version),
                          out_dir / f"soft_labels_{name}_{version}.csv")
    results = [dict(dataset=name, **evaluate(version, t.wet_prob(Xt[(test.dataset == name).values]),
                                             test[test.dataset == name]))
               for name in test.dataset.unique()]
    pd.DataFrame(results).to_csv(out_dir / f"results_{version}.csv", index=False)
    t.save(out_dir / f"{version}.joblib")
    return t, pd.DataFrame(results)


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Teacher update cycle: fit, export soft labels, save.")
    ap.add_argument("--version", required=True, help="e.g. teacher_v2")
    ap.add_argument("--out", default="teacher_out")
    ap.add_argument("--plate", help="dataset_C_v2 folder (cropped with the D masks)")
    ap.add_argument("--gray", help="dataset_D_v2_balanced folder")
    ap.add_argument("--bin", help="dataset_bin folder (format 2, already cropped)")
    ap.add_argument("--feedback", help="hitl.py feedback folder (contains feedback.csv)")
    ap.add_argument("--masks", help="masks folder (default: found inside --gray)")
    ap.add_argument("--n_aug", type=int, default=2)
    a = ap.parse_args(argv)
    masks = a.masks or (find_masks_dir(a.gray) if a.gray else None)
    ds = {}
    if a.plate:
        ds["dataset_C_v2"] = (a.plate, masks, True)
    if a.gray:
        ds["dataset_D_v2_balanced"] = (a.gray, masks, True)
    if a.bin:
        ds["bin"] = (a.bin, None, False)
    if not ds:
        ap.error("give at least one of --plate / --gray / --bin")
    if a.plate and masks is None:
        print("warning: no D masks found; C images will be used uncropped")
    _, res = run_update(ds, a.out, a.version, a.feedback, a.n_aug)
    print(res.to_string())


if __name__ == "__main__":
    _main()
