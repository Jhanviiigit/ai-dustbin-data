"""Export the student to ONNX for TensorRT on the Jetson Nano (speed tests, 224 vs 160 px).

  python student/export_onnx.py --weights student/runs/gray/weights.pt --out student_224.onnx
  python student/export_onnx.py --weights student/runs/gray160/weights.pt --out student_160.onnx

The graph is image -> [dry_prob, wet_prob], with the calibrated temperature already applied.
A <name>.json next to it records the preprocessing the bin code must repeat (RGB, resize to
size x size, ImageNet mean/std), the labels and the version.

On the Nano (not run here):
  trtexec --onnx=student_224.onnx --fp16 --saveEngine=student_224.engine
  tegrastats          # power, while trtexec runs

A model is only meaningful at the size it was trained at (train.py --img_size). Exporting a
224 model at --size 160 gives a speed number but not an accuracy number.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from common import IMG_SIZE, LABELS, MEAN, STD, load_checkpoint


class Calibrated(nn.Module):
    def __init__(self, model, temperature):
        super().__init__()
        self.model, self.temperature = model, temperature

    def forward(self, x):
        return torch.softmax(self.model(x) / self.temperature, dim=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=None, help="input size (default: the size the model was trained at)")
    args = p.parse_args()
    try:
        import onnx  # noqa: F401  (torch.onnx.export needs it)
    except ImportError:
        raise SystemExit("pip install onnx  (needed by export_onnx.py only)")

    model, ckpt = load_checkpoint(args.weights)
    trained = ckpt.get("img_size", IMG_SIZE)
    size = args.size or trained
    if size != trained:
        print(f"note: trained at {trained} px, exporting at {size} px: fine for timing, not for accuracy")
    T = float(ckpt["temperature"])
    if not ckpt.get("calibrated"):
        print("warning: these weights were not calibrated (calibrate.py): the probabilities are raw")
    net = Calibrated(model, T).eval()
    dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(net, dummy, args.out, input_names=["image"], output_names=["probs"],
                      opset_version=17, dynamo=False)
    Path(args.out).with_suffix(".json").write_text(json.dumps({
        "version": ckpt.get("version"), "temperature": T, "size": size, "color": "RGB", "mean": MEAN, "std": STD,
        "outputs": ["dry_prob", "wet_prob"], "labels": LABELS}, indent=2), encoding="utf-8")
    print(f"saved {args.out} (+ .json), input 1x3x{size}x{size}, T={T:.4f}")
