"""Multi-stage training entry point for ECA-LDNet.

Reproduces the 3-stage progressive schedule from the paper:

* Stage 1 - full training, LR 1e-3, warmup-cosine, 60 epochs.
* Stage 2 - fine-tune,    LR 1e-4, 30 epochs.
* Stage 3 - final polish, LR 5e-5, 20 epochs, batch 8, Charbonnier-heavy loss.

Example
-------
    python -m eca_ldnet.train --reside6k /data/RESIDE-6K --its /data/ITS \\
        --out ./runs --stages 1 2 3
"""
from __future__ import annotations

import argparse
import os

from sklearn.model_selection import train_test_split

from . import config as C
from .callbacks import WarmupCosineDecay, make_callbacks
from .data import build_dataset, collect_pairs, find_dir, set_global_seed
from .losses import combined_loss, combined_loss_s3, mae_metric, psnr_metric, ssim_metric
from .model import build_eca_ldnet, load_pretrained


def _gather_training_pairs(reside6k, its):
    pairs = []
    if reside6k:
        pairs += collect_pairs(
            find_dir(reside6k, ["train/hazy", "Train/hazy"]),
            find_dir(reside6k, ["train/GT", "train/gt", "train/clear"]),
            "RESIDE-6K train",
        )
    if its:
        pairs += collect_pairs(
            find_dir(its, ["hazy", "Hazy", "train/hazy"]),
            find_dir(its, ["clear", "GT", "gt", "train/clear", "train/GT"]),
            "ITS indoor",
        )
    return pairs


def train(args):
    import tensorflow as tf

    set_global_seed(C.SEED)
    model_dir = os.path.join(args.out, "models")
    history_dir = os.path.join(args.out, "history")
    for d in (model_dir, history_dir):
        os.makedirs(d, exist_ok=True)

    pairs = _gather_training_pairs(args.reside6k, args.its)
    if not pairs:
        raise SystemExit("No training pairs found. Check --reside6k / --its paths.")
    hazy = [p[0] for p in pairs]
    gt = [p[1] for p in pairs]
    tr_h, va_h, tr_g, va_g = train_test_split(hazy, gt, test_size=0.10, random_state=C.SEED, shuffle=True)
    print(f"Train: {len(tr_h)}  Val: {len(va_h)}")

    model = load_pretrained(args.resume) if args.resume else build_eca_ldnet()
    print(f"ECA-LDNet: {model.count_params():,} params")

    stage_cfgs = {1: C.STAGE1, 2: C.STAGE2, 3: C.STAGE3}
    for s in args.stages:
        cfg = stage_cfgs[s]
        loss_fn = combined_loss_s3 if s == 3 else combined_loss
        train_ds = build_dataset(tr_h, tr_g, augment=True, batch=cfg.batch_size)
        val_ds = build_dataset(va_h, va_g, augment=False, batch=cfg.batch_size, shuffle=False, cache_ds=True)
        steps = max(1, len(tr_h) // cfg.batch_size)
        lr_sched = WarmupCosineDecay(cfg.lr, cfg.warmup_epochs * steps, cfg.epochs * steps)
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_sched, weight_decay=1e-4),
            loss=loss_fn,
            metrics=[psnr_metric, ssim_metric, mae_metric],
        )
        print(f"\n=== Stage {s} | LR={cfg.lr} | {cfg.epochs} epochs | batch {cfg.batch_size} ===")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.epochs,
            callbacks=make_callbacks(s, model_dir, history_dir, patience=C.PATIENCE_ES),
            verbose=1,
        )
    model.save(os.path.join(model_dir, "eca_ldnet_final.keras"))
    print(f"Saved final model to {model_dir}/eca_ldnet_final.keras")


def main():
    p = argparse.ArgumentParser(description="Train ECA-LDNet")
    p.add_argument("--reside6k", default=None)
    p.add_argument("--its", default=None)
    p.add_argument("--out", default="./runs")
    p.add_argument("--resume", default=None, help="Resume from a .keras checkpoint")
    p.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3])
    train(p.parse_args())


if __name__ == "__main__":
    main()
