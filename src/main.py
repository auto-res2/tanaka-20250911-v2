"""src/main.py
Quick-start executable that wires together preprocessing, model construction,
and the training loop. Designed for demonstration/CI rather than large-scale
experiments. All outputs (images + JSON metrics) are written to the mandatory
paths defined in the problem statement.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .preprocess import get_dataloaders
from .train import (
    build_baseline_dit,
    build_ram_dit,
    fixed_seed,
    run_training,
)


CONFIG_PATH = Path("config/config.yaml")


def _load_cfg():  # noqa: D401
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("Missing config/config.yaml – please create one or supply --config path")
    with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def main():  # noqa: D401
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["baseline", "ramdit"], default="baseline")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = _load_cfg()
    fixed_seed(args.seed)

    image_size = cfg.get("image_size", 64)
    train_loader, val_loader = get_dataloaders(args.batch, image_size)

    method_cfg = cfg.get("method_cfg", {})
    if args.method == "baseline":
        model = build_baseline_dit(cfg["model"], image_size, method_cfg)
    else:
        model = build_ram_dit(cfg["model"], image_size, method_cfg)

    arm_cfg = {
        "lr": cfg.get("lr", 1e-4),
        "total_steps": args.steps,
        "eval_every": cfg.get("eval_every", 5),
        "save_every": cfg.get("save_every", 10),
        "samples_eval": cfg.get("samples_eval", 8),
    }
    logdir = Path(".research/iteration2/logs")
    run_training(model, train_loader, val_loader, arm_cfg, logdir)


if __name__ == "__main__":
    main()
