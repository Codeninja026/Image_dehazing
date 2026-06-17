"""Benchmark evaluation: PSNR/SSIM over SOTS / RESIDE-6K + latency/MACs + CLI.

Example
-------
    python -m eca_ldnet.evaluate --model checkpoints/stage5_final_ema_model.keras \\
        --sots /path/to/SOTS --reside6k /path/to/RESIDE-6K --output ./eval_out
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import tensorflow as tf

from .config import IMG_SIZE
from .data import find_dir, get_gt_path, load_image_pair


def _psnr(p, g):
    return float(tf.image.psnr(p[np.newaxis], g[np.newaxis], 1.0).numpy()[0])


def _ssim(p, g):
    return float(tf.image.ssim(p[np.newaxis], g[np.newaxis], 1.0).numpy()[0])


def predict_full(model, img):
    return np.clip(model.predict(img[np.newaxis], verbose=0)[0], 0, 1).astype(np.float32)


def evaluate_dataset(model, hazy_dir, gt_dir, name="", limit=None):
    """Mean PSNR / SSIM over a dataset directory; returns (psnr, ssim) or None."""
    if not hazy_dir or not gt_dir:
        return None
    files = sorted(f for f in os.listdir(hazy_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if limit:
        files = files[:limit]
    psnrs, ssims = [], []
    for fname in files:
        gp = get_gt_path(fname, gt_dir)
        if not gp:
            continue
        hazy, gt = load_image_pair(os.path.join(hazy_dir, fname), gp)
        if hazy is None:
            continue
        pred = predict_full(model, hazy)
        psnrs.append(_psnr(pred, gt))
        ssims.append(_ssim(pred, gt))
    if not psnrs:
        return None
    print(f"  {name:<14}: PSNR={np.mean(psnrs):.4f}  SSIM={np.mean(ssims):.4f}  (n={len(psnrs)})")
    return float(np.mean(psnrs)), float(np.mean(ssims))


def measure_latency_and_flops(model, runs=100, warmup=20):
    """Wall-clock latency / FPS plus a rough MACs estimate for the model."""
    dummy = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    for _ in range(warmup):
        model.predict(dummy, verbose=0)
    t0 = time.time()
    for _ in range(runs):
        model.predict(dummy, verbose=0)
    avg_ms = ((time.time() - t0) / runs) * 1000.0
    params_m = model.count_params() / 1e6
    macs_g = (model.count_params() * IMG_SIZE * IMG_SIZE) / 1e9
    return {"avg_ms": avg_ms, "fps": 1000.0 / avg_ms, "macs_g": macs_g, "params_m": params_m}


def main():
    p = argparse.ArgumentParser(description="ECA-LDNet benchmark evaluation")
    p.add_argument("--model", required=True)
    p.add_argument("--sots", default=None, help="SOTS root (indoor/ + outdoor/)")
    p.add_argument("--reside6k", default=None, help="RESIDE-6K root")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", default="./eval_out")
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    from .model import load_pretrained

    model = load_pretrained(args.model)
    print(f"Model: {model.count_params():,} params")

    lat = measure_latency_and_flops(model)
    print(f"Latency: {lat['avg_ms']:.2f} ms  ({lat['fps']:.2f} FPS)  MACs~{lat['macs_g']:.2f}G")

    if args.sots:
        evaluate_dataset(
            model,
            find_dir(args.sots, ["indoor/hazy", "SOTS/indoor/hazy"]),
            find_dir(args.sots, ["indoor/clear", "indoor/gt", "indoor/GT"]),
            "SOTS-Indoor",
            args.limit,
        )
        evaluate_dataset(
            model,
            find_dir(args.sots, ["outdoor/hazy", "SOTS/outdoor/hazy"]),
            find_dir(args.sots, ["outdoor/clear", "outdoor/Clear", "outdoor/GT"]),
            "SOTS-Outdoor",
            args.limit,
        )
    if args.reside6k:
        evaluate_dataset(
            model,
            find_dir(args.reside6k, ["test/hazy", "Test/hazy"]),
            find_dir(args.reside6k, ["test/GT", "test/gt", "test/clear"]),
            "RESIDE-6K",
            args.limit,
        )


if __name__ == "__main__":
    main()
