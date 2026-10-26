"""Shared constants and helpers for the student model (MobileNetV2, wet/dry waste)."""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

GRAY = 128                 # background level of dataset D (remove_background.py --gray 128)
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet
CLASSES = ["dry waste", "wet waste"]    # index 0 / 1; matches sorted folder names
LABELS = ["dry", "wet"]                 # what classify_image() returns, same order as CLASSES
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Dataset folder for each background style. Same images; only the background differs.
VARIANTS = {"gray": "dataset_D_v2_balanced", "plate": "dataset_C_v2"}
# "bin" = photos captured by the bin's own camera, later. They are already the detector's
# crops (item box + 10% padding, gray 128 outside), so they are not cropped again.
# Same layout as the others: dataset_bin/{train,val,test}/{dry waste,wet waste}/
VARIANTS["bin"] = "dataset_bin"
# Item masks for both variants (same file names). Used to crop every image to its item.
MASKS = "dataset_D_v2_balanced/masks"

# The detector crops the item's box + 10% padding (bin_vision item_crop). Training
# varies the padding a little; val and test use exactly the detector's.
EVAL_PAD = (0.10, 0.10)
TRAIN_PAD = (0.05, 0.30)
MIN_MASK_FRAC = 0.01       # masks covering less than this are failed cut-outs: keep the whole image

def eval_tf(size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_tf(size=IMG_SIZE):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


EVAL_TF, TRAIN_TF = eval_tf(), train_tf()      # the 224 px defaults


def build_model(pretrained):
    """MobileNetV2 with a 2-class head. pretrained=True loads ImageNet weights."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    return model


def bgr_to_pil(img_bgr):
    return Image.fromarray(np.ascontiguousarray(img_bgr[:, :, ::-1]))


def crop_to_item(im, mask, pad):
    """Crop a PIL image to its item the way the detector does: the mask's bounding
    box + pad * (longer side) on every side, clamped to the image.

    The mask may be smaller than the image (masks are stored at max 512 px), so the
    box is scaled to the image size. No usable mask -> the whole image."""
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


class SplitDataset(torch.utils.data.Dataset):
    """Images of one split: <data_dir>/<split>/<class>/<file>. Returns (tensor, label).

    With masks_dir (<masks_dir>/<split>/<class>/<stem>.png), every image is first
    cropped to its item with a padding drawn from pad = (low, high), so training
    and evaluation see the same framing as the detector's crops on the bin.

    extra = [(path, label), ...] adds images from elsewhere (human-labelled bin crops).
    They are the detector's crops already, so they are never cropped again."""

    def __init__(self, data_dir, split, transform, masks_dir=None, pad=EVAL_PAD, seed=0, extra=()):
        from pathlib import Path
        self.transform = transform
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.split = split
        self.pad = pad
        self.rng = np.random.default_rng(seed)
        self.items = []
        for label, cls in enumerate(CLASSES):
            folder = Path(data_dir) / split / cls
            if not folder.is_dir():
                raise FileNotFoundError(f"missing folder: {folder}")
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    self.items.append((p, label))
        if not self.items:
            raise RuntimeError(f"no images under {Path(data_dir) / split}")
        if self.masks_dir is not None:
            missing = sum(not self._mask_path(p, y).exists() for p, y in self.items)
            if missing:
                print(f"warning: {missing} of {len(self.items)} {split} images have no mask; they are used uncropped")
        self.no_crop = {Path(p) for p, _ in extra}
        self.items += [(Path(p), y) for p, y in extra]

    def _mask_path(self, path, label):
        return self.masks_dir / self.split / CLASSES[label] / (path.stem + ".png")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        with Image.open(path) as im:
            im = im.convert("RGB")
        if self.masks_dir is not None and path not in self.no_crop:
            mask_path = self._mask_path(path, label)
            mask = Image.open(mask_path) if mask_path.exists() else None
            lo, hi = self.pad
            im = crop_to_item(im, mask, lo if lo == hi else float(self.rng.uniform(lo, hi)))
        return self.transform(im), label


def _class_folder(value):
    """'dry' / 'wet' / 'dry waste' / 'wet waste' -> 'dry waste' / 'wet waste'."""
    v = str(value).strip().lower()
    for cls in CLASSES:
        if v in (cls, cls.split()[0]):
            return cls
    raise ValueError(f"unknown class {value!r}")


def _stem(name):
    from pathlib import PurePosixPath
    return PurePosixPath(str(name).replace("\\", "/")).stem


class SoftLabels:
    """The teacher's calibrated P(wet) per image, from a soft-label CSV.

    Matched on (split, class folder, file name without extension): two different images
    can share a file name, but not within one split and class. Two layouts are read:
      agreed:   split, class_folder, filename, teacher_wet_prob [, teacher_version]
      notebook: path, split, label (dry/wet), teacher_wet_prob   (teacher/ branch, v0)
    Train soft labels should be out-of-fold; this loader cannot check that."""

    def __init__(self, csv_path):
        import csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise SystemExit(f"{csv_path} is empty")
        cols = set(rows[0])
        if "teacher_wet_prob" not in cols or "split" not in cols:
            raise SystemExit(f"{csv_path}: needs split and teacher_wet_prob; found {sorted(cols)}")
        if {"class_folder", "filename"} <= cols:
            cls_of, name_of = (lambda r: r["class_folder"]), (lambda r: r["filename"])
        elif {"label", "path"} <= cols:
            cls_of, name_of = (lambda r: r["label"]), (lambda r: r["path"])
        else:
            raise SystemExit(f"{csv_path}: needs class_folder+filename or label+path; found {sorted(cols)}")
        self.p, self.duplicates = {}, 0
        for r in rows:
            p = float(r["teacher_wet_prob"])
            if not 0.0 <= p <= 1.0:
                raise SystemExit(f"{csv_path}: teacher_wet_prob {p} is not a probability")
            key = (r["split"].strip().lower(), _class_folder(cls_of(r)), _stem(name_of(r)))
            self.duplicates += key in self.p
            self.p.setdefault(key, p)
        self.versions = sorted({r["teacher_version"] for r in rows if r.get("teacher_version")})
        self.any_split = {}
        for (split, cls, stem), p in self.p.items():
            self.any_split.setdefault((cls, stem), p)

    def get(self, split, label, stem):
        return self.p.get((split, CLASSES[label], stem))

    def get_any_split(self, label, stem):
        """For human-labelled bin crops, whose split in the teacher's file is not fixed."""
        return self.any_split.get((CLASSES[label], stem))


def file_hashes(data_dir, splits=("val", "test")):
    """sha256 of every image in the given splits of a dataset folder (for the leak guard)."""
    import hashlib
    from pathlib import Path
    out = set()
    for split in splits:
        for cls in CLASSES:
            folder = Path(data_dir) / split / cls
            if folder.is_dir():
                out |= {hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS}
    return out


def _box(text):
    """'x y w h' (as hitl.py writes the item's box) -> (x0, y0, x1, y1), or None if unreadable."""
    import re
    n = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", str(text))]
    return (n[0], n[1], n[0] + n[2], n[1] + n[3]) if len(n) == 4 and n[2] > 0 and n[3] > 0 else None


def _overlap(a, b):
    """True if two boxes share some area (boxes that only touch along an edge do not)."""
    return min(a[2], b[2]) - max(a[0], b[0]) > 0 and min(a[3], b[3]) - max(a[1], b[1]) > 0


def load_feedback(root, exclude_hashes=frozenset()):
    """Human-labelled crops from a hitl.py FeedbackStore folder -> [(path, label)].

    Uses <root>/feedback.csv (crop_file is relative to root). Every item the human labelled
    is used, including the items of loads the human called dry+wet: each crop was labelled
    on its own, and those are the loads the student was unsure about, the most useful ones.

    Skipped:
      * items the human skipped (empty human_label), and missing files;
      * a crop whose box overlaps another item's box in the same load (touching items: the
        crop may contain part of its neighbour, so its label is not reliable). Boxes that only
        touch along an edge do not count. A box that cannot be read is kept and counted;
      * a crop whose bytes equal an image in exclude_hashes (the val and test sets must never
        be trained on; this catches exact copies only, not another photo of the same item)."""
    import csv
    import hashlib
    from pathlib import Path
    root = Path(root)
    with open(root / "feedback.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    boxes = {}                                          # load_id -> [(row index, box)] of every item
    for i, r in enumerate(rows):
        b = _box(r.get("box", ""))
        if b is not None:
            boxes.setdefault(r["load_id"], []).append((i, b))
    used = []
    skipped = {"no label": 0, "file missing": 0, "box overlaps another item": 0, "same as val/test image": 0}
    unreadable = 0
    for i, r in enumerate(rows):
        mine = _box(r.get("box", ""))
        if r["human_label"] not in ("dry", "wet"):
            skipped["no label"] += 1
        elif not (root / r["crop_file"]).is_file():
            skipped["file missing"] += 1
        elif mine is not None and any(j != i and _overlap(mine, b) for j, b in boxes.get(r["load_id"], [])):
            skipped["box overlaps another item"] += 1
        elif exclude_hashes and hashlib.sha256((root / r["crop_file"]).read_bytes()).hexdigest() in exclude_hashes:
            skipped["same as val/test image"] += 1
        else:
            unreadable += mine is None
            used.append((root / r["crop_file"], LABELS.index(r["human_label"])))
    mixed = sum(1 for load in {r["load_id"] for r in rows}
                if len({r["human_label"] for r in rows if r["load_id"] == load and r["human_label"] in ("dry", "wet")}) > 1)
    print(f"feedback: {len(rows)} rows, {len(used)} used, skipped {skipped}"
          f"{f', {unreadable} used with an unreadable box' if unreadable else ''}; "
          f"{mixed} dry+wet load(s) kept")
    return used


@torch.no_grad()
def collect_logits(model, dataset, batch_size=64):
    """Returns (logits [N,2], labels [N]) in dataset order."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x))
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
