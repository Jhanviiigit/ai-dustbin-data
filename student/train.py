"""Fine-tune MobileNetV2 (ImageNet weights) on single-item crops.

  python student/train.py --variant gray  --data_root .      -> student/runs/gray/weights.pt
  python student/train.py --variant plate --data_root .      -> student/runs/plate/weights.pt
  python student/train.py --variant bin --init_weights student/runs/gray/weights.pt --lr 3e-5

gray  = dataset_D_v2_balanced (background removed, gray 128)
plate = dataset_C_v2          (real plate)
bin   = dataset_bin           (photos from the bin's camera; already the detector's crops)
Both public variants use the same images, seed and settings, so only the background differs.
Only train/ is used for fitting and val/ for choosing the epoch. test/ is never touched here.

Every public image is cropped to its item first (mask box + padding, as the detector crops
on the bin): random 5-30% padding for train, exactly 10% for val. Masks come from
dataset_D_v2_balanced/masks for both variants. --no_crop trains on whole images.

Distillation (--soft_labels teacher.csv): the loss for an image with a teacher probability is
    distill_alpha * CE(true label)  +  (1 - distill_alpha) * CE(teacher's [P(dry), P(wet)])
Images without a teacher probability use the true label only. Validation (choosing the
epoch) always uses the true labels, so runs with and without distillation are comparable.

Human feedback (--feedback_dir <FeedbackStore folder>): the human-labelled crops are added to
train/ (never val/ or test/).
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (CLASSES, EVAL_PAD, GRAY, IMG_SIZE, LABELS, MASKS, TRAIN_PAD, VARIANTS, SoftLabels,
                    SplitDataset, build_model, collect_logits, eval_tf, file_hashes, load_checkpoint, load_feedback,
                    train_tf)

p = argparse.ArgumentParser()
p.add_argument("--variant", required=True, choices=VARIANTS)
p.add_argument("--data_root", default=".", help="folder that contains the dataset folders")
p.add_argument("--data_dir", default=None, help="override the dataset folder")
p.add_argument("--masks_dir", default=None, help=f"item masks (default: <data_root>/{MASKS})")
p.add_argument("--no_crop", action="store_true", help="train on whole images instead of item crops")
p.add_argument("--init_weights", default=None,
               help="start from this weights.pt instead of ImageNet (fine-tuning the base model on bin photos)")
p.add_argument("--soft_labels", default=None, help="teacher soft-label CSV: train with distillation")
p.add_argument("--distill_alpha", type=float, default=0.5, help="weight of the true-label loss (1 = no distillation)")
p.add_argument("--feedback_dir", default=None, help="FeedbackStore folder: add its human-labelled crops to train/")
p.add_argument("--version", default=None, help="name stored in the weights (default: <variant>-s<seed>)")
p.add_argument("--img_size", type=int, default=None,
               help="input size in pixels (default: the --init_weights model's size, else 224; 160 tests a faster model)")
p.add_argument("--out", default=None, help="default: student/runs/<variant>")
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
if not 0.0 <= args.distill_alpha <= 1.0:
    raise SystemExit("--distill_alpha must be between 0 and 1")

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
data_dir = Path(args.data_dir or Path(args.data_root) / VARIANTS[args.variant])
out_dir = Path(args.out or Path(__file__).parent / "runs" / args.variant)
out_dir.mkdir(parents=True, exist_ok=True)
version = args.version or f"{args.variant}-s{args.seed}"

already_cropped = args.no_crop or args.variant == "bin"     # bin photos are the detector's crops
masks_dir = None if already_cropped else Path(args.masks_dir or Path(args.data_root) / MASKS)
if masks_dir is not None and not masks_dir.is_dir():
    raise SystemExit(f"masks folder not found: {masks_dir} (unzip dataset_D_v2_balanced.zip, or pass --no_crop)")
init_model, init_ckpt = load_checkpoint(args.init_weights) if args.init_weights else (None, {})
img_size = args.img_size or init_ckpt.get("img_size", IMG_SIZE)     # fine-tuning keeps the model's input size
if args.init_weights and img_size != init_ckpt.get("img_size", IMG_SIZE):
    print(f"warning: --init_weights was trained at {init_ckpt.get('img_size', IMG_SIZE)} px, now {img_size} px")
feedback = load_feedback(args.feedback_dir, file_hashes(data_dir)) if args.feedback_dir else []
train_ds = SplitDataset(data_dir, "train", train_tf(img_size), masks_dir, TRAIN_PAD, seed=args.seed, extra=feedback)
val_ds = SplitDataset(data_dir, "val", eval_tf(img_size), masks_dir, EVAL_PAD)
print(f"{args.variant} ({version}): {data_dir}  train={len(train_ds)} (feedback {len(feedback)}) val={len(val_ds)}  "
      f"size {img_size}  crops: {'item box + padding from ' + str(masks_dir) if masks_dir else 'off (whole images)'}")

# Class weights from the train counts, so the rarer class is not ignored.
counts = np.bincount([y for _, y in train_ds.items], minlength=len(CLASSES))
weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)
print("train counts", dict(zip(CLASSES, counts.tolist())), "class weights", weights.tolist())

# Teacher probability of "wet" for every train image (nan = none).
soft = None
if args.soft_labels:
    labels_csv = SoftLabels(args.soft_labels)
    fb_paths = {Path(p) for p, _ in feedback}
    soft = np.full(len(train_ds), np.nan, dtype=np.float32)
    for i, (path, y) in enumerate(train_ds.items):
        q = labels_csv.get_any_split(y, path.stem) if path in fb_paths else labels_csv.get("train", y, path.stem)
        if q is not None:
            soft[i] = q
    have = int(np.isfinite(soft).sum())
    print(f"soft labels: {have} of {len(train_ds)} train images matched"
          f"{f' (teacher {labels_csv.versions})' if labels_csv.versions else ''}"
          f"{f', {labels_csv.duplicates} duplicate keys ignored' if labels_csv.duplicates else ''}")
    if have == 0:
        raise SystemExit("no train image matched the soft-label file (check split / class / file names)")
    if have < 0.5 * len(train_ds):
        print("warning: fewer than half of the train images have a teacher probability")


class WithSoft(torch.utils.data.Dataset):
    def __init__(self, ds, soft):
        self.ds, self.soft = ds, soft

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, y, self.soft[i]


def distill_loss(logits, y, p_wet, alpha):
    """Weighted mean of alpha*CE(true) + (1-alpha)*CE(teacher) over the batch; the class
    weights and the normalisation match nn.CrossEntropyLoss(weight=...), so with alpha=1
    or no teacher probability this is the usual loss."""
    logp = F.log_softmax(logits, 1)
    w = weights[y]
    hard = -logp.gather(1, y[:, None])[:, 0]
    has = ~torch.isnan(p_wet)
    t = torch.stack([1 - p_wet, p_wet], 1).nan_to_num(0.0)
    teacher = -(t * logp).sum(1)
    per = torch.where(has, alpha * hard + (1 - alpha) * teacher, hard)
    return (w * per).sum() / w.sum()


if args.init_weights:
    model = init_model                                 # same MobileNetV2; temperature is re-fitted afterwards
    print(f"starting from {args.init_weights} ({init_ckpt.get('version')})")
else:
    model = build_model(pretrained=True)               # ImageNet
loss_fn = nn.CrossEntropyLoss(weight=weights)          # validation loss, and training loss without distillation
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
loader = torch.utils.data.DataLoader(train_ds if soft is None else WithSoft(train_ds, soft),
                                     batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

best = (-1.0, float("inf"))     # (val accuracy, val loss); higher accuracy wins, then lower loss
for epoch in range(1, args.epochs + 1):
    model.train()
    total = 0.0
    for batch in loader:
        opt.zero_grad()
        if soft is None:
            x, y = batch
            loss = loss_fn(model(x), y)
        else:
            x, y, q = batch
            loss = distill_loss(model(x), y, q, args.distill_alpha)
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
            "calibrated": False,            # calibrate.py sets True; promote.py refuses uncalibrated weights
            "version": version,
            "variant": args.variant,
            "init_from": args.init_weights,   # None = ImageNet
            "soft_labels": args.soft_labels,
            "teacher_version": ",".join(labels_csv.versions) if soft is not None else None,
            "distill_alpha": args.distill_alpha if soft is not None else None,
            "feedback_dir": args.feedback_dir,
            "feedback_used": len(feedback),
            "seed": args.seed,
            "classes": CLASSES,
            "labels": LABELS,
            "masks_dir": str(masks_dir) if masks_dir else None,    # calibrate/evaluate crop the same way
            "gray": GRAY,
            "img_size": img_size,
            "train_images": len(train_ds),
            "soft_matched": int(np.isfinite(soft).sum()) if soft is not None else 0,
            "init_version": init_ckpt.get("version"),
            "epoch": epoch,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "data_dir": str(data_dir),
        }, out_dir / "weights.pt")

print(f"best val acc {best[0]:.4f} (loss {best[1]:.4f}) -> {out_dir / 'weights.pt'}")
