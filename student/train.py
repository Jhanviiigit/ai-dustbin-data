"""Fine-tune MobileNetV2 (ImageNet weights) on single-item crops.

  python student/train.py --variant gray  --data_root .      -> student/runs/gray/weights.pt
  python student/train.py --variant plate --data_root .      -> student/runs/plate/weights.pt

gray  = dataset_D_v2_balanced (background removed, gray 128)
plate = dataset_C_v2          (real plate)
Both use the same images, seed and settings, so only the background differs.
Only train/ is used for fitting and val/ for choosing the epoch. test/ is never touched here.

Every image is cropped to its item first (mask box + padding, as the detector crops
on the bin): random 5-30% padding for train, exactly 10% for val. Masks come from
dataset_D_v2_balanced/masks for both variants. --no_crop trains on whole images.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import (CLASSES, EVAL_PAD, GRAY, IMG_SIZE, LABELS, MASKS, TRAIN_PAD, TRAIN_TF, EVAL_TF,
                    VARIANTS, SplitDataset, build_model, collect_logits, load_checkpoint)

p = argparse.ArgumentParser()
p.add_argument("--variant", required=True, choices=VARIANTS)
p.add_argument("--data_root", default=".", help="folder that contains the dataset folders")
p.add_argument("--data_dir", default=None, help="override the dataset folder")
p.add_argument("--masks_dir", default=None, help=f"item masks (default: <data_root>/{MASKS})")
p.add_argument("--no_crop", action="store_true", help="train on whole images instead of item crops")
p.add_argument("--init_weights", default=None,
               help="start from this weights.pt instead of ImageNet (fine-tuning the base model on bin photos)")
p.add_argument("--out", default=None, help="default: student/runs/<variant>")
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
data_dir = Path(args.data_dir or Path(args.data_root) / VARIANTS[args.variant])
out_dir = Path(args.out or Path(__file__).parent / "runs" / args.variant)
out_dir.mkdir(parents=True, exist_ok=True)

already_cropped = args.no_crop or args.variant == "bin"     # bin photos are the detector's crops
masks_dir = None if already_cropped else Path(args.masks_dir or Path(args.data_root) / MASKS)
if masks_dir is not None and not masks_dir.is_dir():
    raise SystemExit(f"masks folder not found: {masks_dir} (unzip dataset_D_v2_balanced.zip, or pass --no_crop)")
train_ds = SplitDataset(data_dir, "train", TRAIN_TF, masks_dir, TRAIN_PAD, seed=args.seed)
val_ds = SplitDataset(data_dir, "val", EVAL_TF, masks_dir, EVAL_PAD)
print(f"{args.variant}: {data_dir}  train={len(train_ds)} val={len(val_ds)}  "
      f"crops: {'item box + padding from ' + str(masks_dir) if masks_dir else 'off (whole images)'}")

# Class weights from the train counts, so the rarer class is not ignored.
counts = np.bincount([y for _, y in train_ds.items], minlength=len(CLASSES))
weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)
print("train counts", dict(zip(CLASSES, counts.tolist())), "class weights", weights.tolist())

if args.init_weights:
    model, _ = load_checkpoint(args.init_weights)      # same MobileNetV2; temperature is re-fitted afterwards
    print(f"starting from {args.init_weights}")
else:
    model = build_model(pretrained=True)               # ImageNet
loss_fn = nn.CrossEntropyLoss(weight=weights)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                     num_workers=0, drop_last=True)

best = (-1.0, float("inf"))     # (val accuracy, val loss); higher accuracy wins, then lower loss
for epoch in range(1, args.epochs + 1):
    model.train()
    total = 0.0
    for x, y in loader:
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item() * len(y)
    sched.step()

    logits, labels = collect_logits(model, val_ds)
    val_acc = (logits.argmax(1) == labels).float().mean().item()
    val_loss = loss_fn(logits, labels).item()
    print(f"epoch {epoch:2d}  train loss {total / (len(loader) * args.batch_size):.4f}  "
          f"val loss {val_loss:.4f}  val acc {val_acc:.4f}")

    if (val_acc, -val_loss) > (best[0], -best[1]):
        best = (val_acc, val_loss)
        torch.save({
            "state_dict": model.state_dict(),
            "temperature": 1.0,             # set by calibrate.py
            "variant": args.variant,
            "init_from": args.init_weights,   # None = ImageNet
            "classes": CLASSES,
            "labels": LABELS,
            "masks_dir": str(masks_dir) if masks_dir else None,    # calibrate/evaluate crop the same way
            "gray": GRAY,
            "img_size": IMG_SIZE,
            "epoch": epoch,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "data_dir": str(data_dir),
        }, out_dir / "weights.pt")

print(f"best val acc {best[0]:.4f} (loss {best[1]:.4f}) -> {out_dir / 'weights.pt'}")
