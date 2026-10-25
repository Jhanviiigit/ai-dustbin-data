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
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Dataset folder for each background style. Same images; only the background differs.
VARIANTS = {"gray": "dataset_D_v2_balanced", "plate": "dataset_C_v2"}

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


class SplitDataset(torch.utils.data.Dataset):
    """Images of one split: <data_dir>/<split>/<class>/<file>. Returns (tensor, label)."""

    def __init__(self, data_dir, split, transform):
        from pathlib import Path
        self.transform = transform
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

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        with Image.open(path) as im:
            return self.transform(im.convert("RGB")), label


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
