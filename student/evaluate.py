"""Test-split results: accuracy overall and per source, confusion matrix, calibration.

  python student/evaluate.py --weights student/runs/gray/weights.pt
  python student/evaluate.py --weights student/runs/gray/weights.pt --data_dir dataset_C_v2   # cross-check

Run this only for final numbers. Writes results.md next to the weights (or --out).
"""
import argparse
import csv
from pathlib import Path, PurePosixPath

import torch
import torch.nn.functional as F

from calibrate import ece
from common import CLASSES, EVAL_PAD, EVAL_TF, SplitDataset, collect_logits, load_checkpoint


def load_sources(data_dir):
    """Map file stem -> source for the test split, from manifest.csv.

    The file names inside the dataset come from the manifest's new_path column
    ('path' is the original file before cleaning, and RealWaste files were renamed).
    The manifest was written on Windows, so backslashes are normalised first:
    on Linux, Path('test\\\\dry waste\\\\x.jpg').stem would be the whole string."""
    with open(Path(data_dir) / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = rows[0].keys()
    if not {"new_path", "source", "split"} <= set(cols):
        raise SystemExit(f"manifest.csv needs new_path, source and split columns; found {list(cols)}")
    return {PurePosixPath(r["new_path"].replace("\\", "/")).stem: r["source"] for r in rows if r["split"] == "test"}


def report(title, preds, labels, probs, sources):
    n = len(labels)
    lines = [f"### {title}", "", f"Test images: {n}  |  accuracy **{(preds == labels).float().mean():.4f}**"
             f"  |  ECE {ece(probs, labels):.4f}  |  NLL {F.nll_loss(probs.log(), labels):.4f}", ""]
    lines += ["| Source | Images | Accuracy |", "|---|---|---|"]
    for s in sorted(set(sources)):
        m = torch.tensor([x == s for x in sources])
        lines.append(f"| {s} | {int(m.sum())} | {(preds[m] == labels[m]).float().mean():.4f} |")
    lines += ["", "Confusion matrix (rows = true, columns = predicted):", "",
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
    p.add_argument("--out", default=None)
    args = p.parse_args()

    model, ckpt = load_checkpoint(args.weights)
    data_dir = args.data_dir or ckpt["data_dir"]
    T = float(ckpt["temperature"])
    ds = SplitDataset(data_dir, "test", EVAL_TF, args.masks_dir or ckpt.get("masks_dir"), EVAL_PAD)
    logits, labels = collect_logits(model, ds)
    sources_by_stem = load_sources(data_dir)
    sources = [sources_by_stem.get(path.stem, "unknown") for path, _ in ds.items]
    if "unknown" in sources:
        print(f"warning: {sources.count('unknown')} test images not found in manifest.csv")

    preds = logits.argmax(1)
    text = [f"## {ckpt['variant']} model on {Path(data_dir).name}", "",
            f"Weights `{args.weights}`, temperature T = {T:.4f}", "",
            report("Calibrated (as deployed)", preds, labels, F.softmax(logits / T, 1), sources), "",
            report("Uncalibrated (T = 1), for comparison", preds, labels, F.softmax(logits, 1), sources)]
    text = "\n".join(text) + "\n"
    print(text)
    out = Path(args.out) if args.out else Path(args.weights).with_name(f"results_{Path(data_dir).name}.md")
    out.write_text(text, encoding="utf-8")
    print(f"saved {out}")
