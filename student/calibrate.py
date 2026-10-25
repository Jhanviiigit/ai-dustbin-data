"""Temperature scaling on the val split.

  python student/calibrate.py --weights student/runs/gray/weights.pt

Finds the single value T that minimises the val NLL of softmax(logits / T), writes it
into the checkpoint and to temperature.json next to it. T > 1 means the raw model was
over-confident. Accuracy and the predicted label never change; only the confidence does.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common import EVAL_TF, SplitDataset, collect_logits, load_checkpoint


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


def fit_temperature(logits, labels):
    log_t = torch.zeros(1, requires_grad=True)          # T = exp(log_t) stays positive
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.exp().item()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data_dir", default=None, help="default: the dataset the model was trained on")
    args = p.parse_args()

    model, ckpt = load_checkpoint(args.weights)
    val_ds = SplitDataset(args.data_dir or ckpt["data_dir"], "val", EVAL_TF)
    logits, labels = collect_logits(model, val_ds)

    T = fit_temperature(logits, labels)
    before = F.softmax(logits, 1)
    after = F.softmax(logits / T, 1)
    print(f"val images {len(val_ds)}   temperature T = {T:.4f}")
    print(f"NLL {F.nll_loss(before.log(), labels):.4f} -> {F.nll_loss(after.log(), labels):.4f}")
    print(f"ECE {ece(before, labels):.4f} -> {ece(after, labels):.4f}")

    ckpt["temperature"] = T
    torch.save(ckpt, args.weights)
    out = Path(args.weights).with_name("temperature.json")
    out.write_text(json.dumps({"temperature": T, "variant": ckpt["variant"],
                               "val_images": len(val_ds)}, indent=2))
    print(f"saved T into {args.weights} and {out}")
