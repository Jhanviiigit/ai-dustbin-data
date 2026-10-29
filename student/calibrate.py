"""Temperature scaling on the val split.

  python student/calibrate.py --weights student/runs/gray/weights.pt

Finds the single value T that minimises the val NLL of softmax(logits / T), writes it
into the checkpoint and to temperature.json next to it. T > 1 means the raw model was
over-confident. Accuracy and the predicted label never change; only the confidence does.

If the fit lands on the edge of the search range (val has too few errors, or is perfectly
classified, so the NLL keeps falling as T -> 0), it is not a calibration. With
--fallback_from <weights.pt> the temperature of that model (the student being replaced) is
kept instead, and the weights record that; without it the edge value is written, with a warning.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common import EVAL_PAD, IMG_SIZE, SplitDataset, collect_logits, eval_tf, load_checkpoint


def ece(probs, labels, bins=15):
    """Expected calibration error on the top-class confidence."""
    conf, pred = probs.max(1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    total = torch.zeros(())
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean() * (correct[m].mean() - conf[m].mean()).abs()
    return total.item()


T_MIN, T_MAX = 0.05, 20.0


def fit_temperature(logits, labels, steps=400):
    """The T in [T_MIN, T_MAX] that minimises the NLL of softmax(logits / T).

    A bounded grid search, not gradient descent: on a small or perfectly separable val set
    the NLL keeps falling as T -> 0, and an unbounded optimiser runs off to 0 or NaN."""
    temps = torch.logspace(torch.log10(torch.tensor(T_MIN)).item(), torch.log10(torch.tensor(T_MAX)).item(), steps)
    nll = torch.stack([F.cross_entropy(logits / t, labels) for t in temps])
    return temps[nll.argmin()].item()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data_dir", default=None, help="default: the dataset the model was trained on")
    p.add_argument("--masks_dir", default=None, help="default: the masks the model was trained with (item crops)")
    p.add_argument("--no_crop", action="store_true",
                   help="images are already crops (e.g. dataset_bin): do not cut them with masks")
    p.add_argument("--fallback_from", default=None,
                   help="weights whose temperature is kept when the fit lands on the edge of the search range")
    args = p.parse_args()

    model, ckpt = load_checkpoint(args.weights)
    masks_dir = None if args.no_crop else (args.masks_dir or ckpt.get("masks_dir"))
    data_dir = args.data_dir or ckpt["data_dir"]
    val_ds = SplitDataset(data_dir, "val", eval_tf(ckpt.get("img_size", IMG_SIZE)), masks_dir, EVAL_PAD)
    logits, labels = collect_logits(model, val_ds)

    T = fit_temperature(logits, labels)
    note = None
    print(f"val images {len(val_ds)}   temperature T = {T:.4f}")
    if T <= T_MIN * 1.02 or T >= T_MAX / 1.02:
        print(f"warning: T hit the search limit [{T_MIN}, {T_MAX}]: the val set is too small or too easy "
              f"for a reliable calibration")
        if args.fallback_from:
            kept = float(load_checkpoint(args.fallback_from)[1]["temperature"])
            note = f"the val fit hit the search limit (T = {T:.4f}); kept T = {kept:.4f} of {args.fallback_from}"
            print(note)
            T = kept
    before = F.softmax(logits, 1)
    after = F.softmax(logits / T, 1)
    print(f"NLL {F.nll_loss(before.log(), labels):.4f} -> {F.nll_loss(after.log(), labels):.4f}")
    print(f"ECE {ece(before, labels):.4f} -> {ece(after, labels):.4f}")

    ckpt["temperature"] = T
    ckpt["calibrated"] = True
    ckpt["calibrated_on"] = str(data_dir) + (f" ({note})" if note else "")   # which val split T was fitted on
    ckpt["calibration_note"] = note
    ckpt["calibration_images"] = len(val_ds)
    torch.save(ckpt, args.weights)
    out = Path(args.weights).with_name("temperature.json")
    out.write_text(json.dumps({"temperature": T, "variant": ckpt["variant"], "version": ckpt.get("version"),
                               "calibrated_on": str(data_dir), "val_images": len(val_ds)}, indent=2))
    print(f"saved T into {args.weights} and {out}")
