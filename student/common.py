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

EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


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
    and evaluation see the same framing as the detector's crops on the bin."""

    def __init__(self, data_dir, split, transform, masks_dir=None, pad=EVAL_PAD, seed=0):
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

    def _mask_path(self, path, label):
        return self.masks_dir / self.split / CLASSES[label] / (path.stem + ".png")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        with Image.open(path) as im:
            im = im.convert("RGB")
        if self.masks_dir is not None:
            mask_path = self._mask_path(path, label)
            mask = Image.open(mask_path) if mask_path.exists() else None
            lo, hi = self.pad
            im = crop_to_item(im, mask, lo if lo == hi else float(self.rng.uniform(lo, hi)))
        return self.transform(im), label


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
