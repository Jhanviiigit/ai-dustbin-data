"""Test-split results: accuracy overall and per source, confusion matrix, calibration.

  python student/evaluate.py --weights student/runs/gray/weights.pt
  python student/evaluate.py --weights student/runs/gray/weights.pt --data_dir dataset_C_v2   # cross-check
  python student/evaluate.py --weights student/runs/gray/weights.pt --data_dir dataset_bin --no_crop

Run this only for final numbers. Writes results_<dataset>.md next to the weights (or --out).
metrics() is also used by promote.py and experiments.py.
"""
import argparse
import csv
from pathlib import Path, PurePosixPath

import torch
import torch.nn.functional as F

from calibrate import ece
from common import CLASSES, EVAL_PAD, IMG_SIZE, SplitDataset, collect_logits, eval_tf, load_checkpoint

OPTIONAL_GROUPS = ("lighting",)       # extra manifest columns that get their own accuracy table


def load_groups(data_dir, split="test"):
    """{column: {file stem: value}} for the split, from manifest.csv: 'source', plus the
    optional columns above when present.

    The file names inside the dataset come from the manifest's new_path column
    ('path' is the original file before cleaning, and RealWaste files were renamed).
    The manifest was written on Windows, so backslashes are normalised first:
    on Linux, Path('test\\\\dry waste\\\\x.jpg').stem would be the whole string."""
    with open(Path(data_dir) / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = set(rows[0].keys())
    if {"new_path", "source", "split"} <= cols:
        path_col, source_col = "new_path", "source"
    elif {"path", "origin", "split"} <= cols:               # the shared pool (update/pool.py): crop_id.jpg, origin
        path_col, source_col = "path", "origin"
    else:
        raise SystemExit(f"manifest.csv needs new_path, source and split columns (or path, origin, split); "
                         f"found {sorted(cols)}")
    columns = {"source": source_col, **{c: c for c in OPTIONAL_GROUPS if c in cols}}
    return {name: {PurePosixPath(r[path_col].replace("\\", "/")).stem: r[col] for r in rows if r["split"] == split}
            for name, col in columns.items()}


def metrics(weights, data_dir=None, split="test", no_crop=False, masks_dir=None):
    """Evaluate a weights file on one split. Returns a dict with the calibrated accuracy,
    NLL and ECE, the per-image tensors, and the per-group values."""
    model, ckpt = load_checkpoint(weights)
    data_dir = data_dir or ckpt["data_dir"]
    T = float(ckpt["temperature"])
    masks = None if no_crop else (masks_dir or ckpt.get("masks_dir"))
    ds = SplitDataset(data_dir, split, eval_tf(ckpt.get("img_size", IMG_SIZE)), masks, EVAL_PAD)
    logits, labels = collect_logits(model, ds)
    groups = {}
    if (Path(data_dir) / "manifest.csv").exists():
        for col, by_stem in load_groups(data_dir, split).items():
            groups[col] = [by_stem.get(path.stem, "unknown") for path, _ in ds.items]
    else:                       # no manifest: one group
        groups["source"] = ["all"] * len(ds)
    probs = F.softmax(logits / T, 1)
    preds = logits.argmax(1)
    return {"ckpt": ckpt, "data_dir": str(data_dir), "T": T, "n": len(labels), "logits": logits, "labels": labels,
            "preds": preds, "probs": probs, "groups": groups,
            "acc": (preds == labels).float().mean().item(),
            "nll": F.nll_loss(probs.clamp_min(1e-12).log(), labels).item(), "ece": ece(probs, labels)}


def report(title, preds, labels, probs, groups, split="test"):
    n = len(labels)
    lines = [f"### {title}", "", f"{split.capitalize()} images: {n}  |  accuracy **{(preds == labels).float().mean():.4f}**"
             f"  |  ECE {ece(probs, labels):.4f}  |  NLL {F.nll_loss(probs.clamp_min(1e-12).log(), labels):.4f}", ""]
    for name, values in groups.items():
        lines += [f"| {name.capitalize()} | Images | Accuracy |", "|---|---|---|"]
        for s in sorted(set(values)):
            m = torch.tensor([x == s for x in values])
            lines.append(f"| {s} | {int(m.sum())} | {(preds[m] == labels[m]).float().mean():.4f} |")
        lines.append("")
    lines += ["Confusion matrix (rows = true, columns = predicted):", "",
              "| | " + " | ".join(f"pred {c}" for c in CLASSES) + " |", "|---|---|---|"]
    for t, cls in enumerate(CLASSES):
        row = [int(((labels == t) & (preds == p)).sum()) for p in range(len(CLASSES))]
        lines.append(f"| true {cls} | " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data_dir", default=None, help="default: the dataset the model was trained on")
    p.add_argument("--masks_dir", default=None, help="default: the masks the model was trained with (item crops)")
    p.add_argument("--no_crop", action="store_true",
                   help="images are already crops (e.g. dataset_bin): do not cut them with masks")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    r = metrics(args.weights, args.data_dir, args.split, args.no_crop, args.masks_dir)
    ckpt, data_dir, T = r["ckpt"], r["data_dir"], r["T"]
    for col, values in r["groups"].items():
        if "unknown" in values:
            print(f"warning: {values.count('unknown')} {args.split} images not found in manifest.csv ({col})")

    text = [f"## {ckpt['variant']} model ({ckpt.get('version')}) on {Path(data_dir).name}", "",
            f"Weights `{args.weights}`, temperature T = {T:.4f}, calibrated: {ckpt.get('calibrated', False)}", "",
            report("Calibrated (as deployed)", r["preds"], r["labels"], r["probs"], r["groups"], args.split), "",
            report("Uncalibrated (T = 1), for comparison", r["preds"], r["labels"], F.softmax(r["logits"], 1),
                   r["groups"], args.split)]
    text = "\n".join(text) + "\n"
    print(text)
    suffix = "" if args.split == "test" else f"_{args.split}"
    out = Path(args.out) if args.out else Path(args.weights).with_name(f"results_{Path(data_dir).name}{suffix}.md")
    out.write_text(text, encoding="utf-8")
    print(f"saved {out}")
