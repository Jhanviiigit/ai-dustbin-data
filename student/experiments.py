"""Public data vs bin data: three ways to train, several seeds each, scored on the bin.

  python student/experiments.py --data_root . --public_variant gray --seeds 1 2 3

  public_only    trained on the public dataset (C or D) only
  bin_only       trained on dataset_bin only (from ImageNet)
  public_to_bin  the public model, then fine-tuned on dataset_bin (--init_weights)

Every model is calibrated on dataset_bin/val and scored on dataset_bin/test (already crops,
so no mask cropping). Output: <out>/summary.md (mean and std over seeds) and results.json.
Run it once dataset_bin exists. Each seed trains three models, so this is slow on a CPU.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from common import VARIANTS
from evaluate import metrics

HERE = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--data_root", default=".", help="folder that contains the dataset folders, incl. dataset_bin")
p.add_argument("--public_variant", default="gray", choices=["gray", "plate"])
p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--finetune_lr", type=float, default=3e-5)
p.add_argument("--no_crop_public", action="store_true", help="pass --no_crop when training the public model")
p.add_argument("--out", default=str(HERE / "runs" / "experiments"))
args = p.parse_args()

root, out = Path(args.data_root), Path(args.out)
bin_dir = root / VARIANTS["bin"]


def run(script, *a):
    cmd = [sys.executable, str(HERE / script), *map(str, a)]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def train(variant, seed, cfg, *extra):
    d = out / cfg / f"seed{seed}"
    run("train.py", "--variant", variant, "--data_root", root, "--seed", seed, "--epochs", args.epochs,
        "--out", d, "--version", f"{cfg}-s{seed}", *extra)
    return d / "weights.pt"


results = {}                                    # config -> [metrics per seed]
for seed in args.seeds:
    pub = train(args.public_variant, seed, "public_only", *(["--no_crop"] if args.no_crop_public else []))
    run("calibrate.py", "--weights", pub, "--data_dir", bin_dir, "--no_crop")       # T on bin val
    only = train("bin", seed, "bin_only")
    run("calibrate.py", "--weights", only)
    both = train("bin", seed, "public_to_bin", "--init_weights", pub, "--lr", args.finetune_lr)
    run("calibrate.py", "--weights", both)
    for cfg, w in (("public_only", pub), ("bin_only", only), ("public_to_bin", both)):
        m = metrics(w, bin_dir, "test", no_crop=True)
        results.setdefault(cfg, []).append({"seed": seed, **{k: m[k] for k in ("acc", "nll", "ece", "T", "n")}})


def stat(cfg, key):
    v = [r[key] for r in results[cfg]]
    return f"{statistics.mean(v):.4f} ± {statistics.stdev(v):.4f}" if len(v) > 1 else f"{v[0]:.4f}"


lines = [f"# Public vs bin data (public = {args.public_variant}), scored on dataset_bin/test", "",
         f"Seeds {args.seeds}, {args.epochs} epochs, fine-tune lr {args.finetune_lr}. "
         f"Mean ± std over seeds ({'std needs 2+ seeds' if len(args.seeds) < 2 else 'sample std'}).", "",
         "| Training data | Accuracy | NLL | ECE |", "|---|---|---|---|"]
for cfg in results:
    lines.append(f"| {cfg} | {stat(cfg, 'acc')} | {stat(cfg, 'nll')} | {stat(cfg, 'ece')} |")
lines += ["", "Per seed (accuracy):", "", "| Training data | " + " | ".join(f"seed {s}" for s in args.seeds) + " |",
          "|---|" + "---|" * len(args.seeds)]
for cfg, rs in results.items():
    lines.append(f"| {cfg} | " + " | ".join(f"{r['acc']:.4f}" for r in rs) + " |")
lines += ["", f"Test images: {results['bin_only'][0]['n']}."]
out.mkdir(parents=True, exist_ok=True)
(out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\n" + "\n".join(lines))
