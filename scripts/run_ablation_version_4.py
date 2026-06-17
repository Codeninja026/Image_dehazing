#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Turnkey controlled ablation + perceptual-metric study for ECA-LDNet.

Run this where the RESIDE datasets and a GPU are available (e.g. Kaggle). It
trains the full model and three knock-out variants under an *identical* protocol
and reports PSNR / SSIM / MS-SSIM (and LPIPS if installed) so the marginal value
of each component (ECA channel attention, pixel attention, physics guidance) can
be quantified. It also evaluates the released checkpoint with the perceptual
metrics that were missing from the paper.

This version supports continuation training for the Full model from a saved
checkpoint, while keeping the ablation variants trained from scratch.

Example (Kaggle paths):
    python scripts/run_ablation_v2.py \
        --reside6k /kaggle/input/.../RESIDE-6K \
        --its /kaggle/input/.../ITS \
        --sots /kaggle/input/.../SOTS \
        --epochs 40 --out /kaggle/working/ablation_out

Notes
-----
* ``--epochs`` is the final epoch target for each variant.
* ``--resume`` is used only for the Full (ECA+PA+Physics) variant.
* ``--pretrained`` additionally evaluates a released .keras checkpoint with
  PSNR/SSIM/MS-SSIM/LPIPS (no training) so the perceptual metrics can be added
  to the paper directly.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from eca_ldnet import config as C  # noqa: E402
from eca_ldnet.data import (  # noqa: E402
    build_dataset,
    collect_pairs,
    find_dir,
    get_gt_path,
    load_image_pair,
    set_global_seed,
)
from eca_ldnet.metrics import lpips_available, lpips_distance, ms_ssim, psnr, ssim  # noqa: E402

VARIANTS = [
    # ("Full (ECA+PA+Physics)", dict()),
    # ("w/o ECA", dict(use_eca=False)),
    # ("w/o Pixel Attention", dict(use_pa=False)),
    ("w/o Physics guidance", dict(use_physics=False)),
    ("w/o all attention", dict(use_eca=False, use_pa=False)),
]


def _gather_train(reside6k, its):
    pairs = []
    if reside6k:
        pairs += collect_pairs(
            find_dir(reside6k, ["train/hazy", "Train/hazy"]),
            find_dir(reside6k, ["train/GT", "train/gt", "train/clear"]),
            "RESIDE-6K",
        )
    if its:
        pairs += collect_pairs(
            find_dir(its, ["hazy", "Hazy", "train/hazy"]),
            find_dir(its, ["clear", "GT", "gt", "train/clear", "train/GT"]),
            "ITS",
        )
    return pairs


def _eval_dir(model, hazy_dir, gt_dir, limit=None):
    if not hazy_dir or not gt_dir:
        return None

    files = sorted(
        f for f in os.listdir(hazy_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if limit:
        files = files[:limit]

    P, S, M, L = [], [], [], []
    use_lpips = lpips_available()

    for f in files:
        gp = get_gt_path(f, gt_dir)
        if not gp:
            continue

        hazy, gt = load_image_pair(os.path.join(hazy_dir, f), gp)
        if hazy is None:
            continue

        pred = np.clip(model.predict(hazy[None], verbose=0)[0], 0, 1).astype("float32")
        P.append(psnr(pred, gt))
        S.append(ssim(pred, gt))
        M.append(ms_ssim(pred, gt))
        if use_lpips:
            L.append(lpips_distance(pred, gt))

    if not P:
        return None

    out = {
        "psnr": float(np.mean(P)),
        "ssim": float(np.mean(S)),
        "ms_ssim": float(np.mean(M)),
        "n": len(P),
    }
    if L:
        out["lpips"] = float(np.mean(L))
    return out


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve_existing_path(path_str: str | None) -> str | None:
    """
    Resolve a checkpoint path from:
      1) the exact path given
      2) the current working directory
      3) the script directory
    """
    if not path_str:
        return None

    p = Path(path_str)
    if p.exists():
        return str(p)

    cwd_candidate = Path.cwd() / path_str
    if cwd_candidate.exists():
        return str(cwd_candidate)

    script_candidate = _script_dir() / path_str
    if script_candidate.exists():
        return str(script_candidate)

    return None


def _train_one(flags, tr, va, epochs, batch, out_dir, label, resume_path=None, resume_epoch=0):
    import tensorflow as tf
    from eca_ldnet.callbacks import WarmupCosineDecay
    from eca_ldnet.losses import combined_loss, mae_metric, psnr_metric, ssim_metric
    from eca_ldnet.model import build_eca_ldnet

    tf.keras.backend.clear_session()

    steps = max(1, len(tr[0]) // batch)
    lr = WarmupCosineDecay(C.LR_S1, 3 * steps, epochs * steps)

    # Resume only the Full model from a saved checkpoint.
    if label == "Full (ECA+PA+Physics)" and resume_path is not None:
        resume_path = _resolve_existing_path(resume_path)

    if label == "Full (ECA+PA+Physics)" and resume_path is not None:
        print(f"\nLoading checkpoint weights for continuation: {resume_path}")

        # Build architecture first
        model = build_eca_ldnet(
            name="ablation",
            **flags
        )

        # Load weights only
        model.load_weights(resume_path)

        print("Checkpoint weights loaded successfully.")
    else:
        print(f"\nCreating fresh model for: {label}")
        model = build_eca_ldnet(name="ablation", **flags)

    # Compile for BOTH cases
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
        loss=combined_loss,
        metrics=[psnr_metric, ssim_metric, mae_metric],
    )

    train_ds = build_dataset(tr[0], tr[1], augment=True, batch=batch)
    val_ds = build_dataset(
        va[0],
        va[1],
        augment=False,
        batch=batch,
        shuffle=False,
        cache_ds=True,
    )

    safe_label = label.replace("/", "_").replace(" ", "_")
    ckpt_path = os.path.join(out_dir, f"{safe_label}_epoch_" + "{epoch:02d}.keras")

    # Save every 7 epochs
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=False,
        verbose=1,
        save_freq=7 * steps,
    )

    # Start from resume epoch only for the Full model
    initial_epoch = resume_epoch if (label == "Full (ECA+PA+Physics)" and resume_path is not None) else 0

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=35,#################################################################################################
        verbose=2,
        callbacks=[ckpt_cb],
    )

    return model


def main():
    p = argparse.ArgumentParser(description="ECA-LDNet ablation + perceptual metrics")
    p.add_argument("--reside6k")
    p.add_argument("--its")
    p.add_argument("--sots")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="cap test images (debug)")
    p.add_argument("--pretrained", default=None, help="release .keras to eval with perceptual metrics")
    p.add_argument("--out", default="./ablation_out")

    # Continuation settings for the Full model
    p.add_argument(
        "--resume",
        default="Full_(ECA+PA+Physics)_epoch_35.keras", ##################################################################
        help="checkpoint to continue the Full model from",
    )
    p.add_argument(
        "--resume-epoch",
        type=int,
        default=28,
        help="initial epoch for the resumed Full model",
    )

    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_global_seed(C.SEED)

    from sklearn.model_selection import train_test_split
    from eca_ldnet.model import build_eca_ldnet

    pairs = _gather_train(args.reside6k, args.its)
    if not pairs:
        raise SystemExit("No training pairs found; check --reside6k / --its.")

    hz, gt = [p[0] for p in pairs], [p[1] for p in pairs]
    tr_h, va_h, tr_g, va_g = train_test_split(hz, gt, test_size=0.10, random_state=C.SEED)

    sin_h = find_dir(args.sots, ["indoor/hazy", "SOTS/indoor/hazy"]) if args.sots else None
    sin_g = find_dir(args.sots, ["indoor/clear", "indoor/gt", "indoor/GT"]) if args.sots else None
    r6_h = find_dir(args.reside6k, ["test/hazy", "Test/hazy"]) if args.reside6k else None
    r6_g = find_dir(args.reside6k, ["test/GT", "test/gt", "test/clear"]) if args.reside6k else None

    rows = []

    resume_path = _resolve_existing_path(args.resume)
    if resume_path:
        print(f"\nContinuation checkpoint found: {resume_path}")
    else:
        print("\nNo continuation checkpoint found; Full model will train from scratch.")

    for label, flags in VARIANTS:
        nparams = build_eca_ldnet(name="count", **flags).count_params()
        print(f"\n===== Training variant: {label}  ({nparams:,} params) =====")

        model = _train_one(
            flags,
            (tr_h, tr_g),
            (va_h, va_g),
            args.epochs,
            args.batch,
            args.out,
            label,
            resume_path=resume_path,
            resume_epoch=args.resume_epoch,
        )

        row = {"variant": label, "params": nparams}
        ind = _eval_dir(model, sin_h, sin_g, args.limit)
        r6k = _eval_dir(model, r6_h, r6_g, args.limit)

        if ind:
            row.update({f"sots_indoor_{k}": v for k, v in ind.items()})
        if r6k:
            row.update({f"reside6k_{k}": v for k, v in r6k.items()})

        rows.append(row)
        print(f"  {label}: SOTS-Indoor={ind}  RESIDE-6K={r6k}")

    if args.pretrained:
        from eca_ldnet.model import load_pretrained

        m = load_pretrained(args.pretrained)
        print("\n===== Released checkpoint perceptual metrics =====")
        rows.append(
            {
                "variant": "Released (paper) checkpoint",
                "params": m.count_params(),
                **{f"sots_indoor_{k}": v for k, v in (_eval_dir(m, sin_h, sin_g, args.limit) or {}).items()},
                **{f"reside6k_{k}": v for k, v in (_eval_dir(m, r6_h, r6_g, args.limit) or {}).items()},
            }
        )

    keys = sorted({k for r in rows for k in r})
    csv_path = os.path.join(args.out, "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved {csv_path}")

    # Human-readable summary
    for r in rows:
        si = r.get("sots_indoor_psnr")
        ss = r.get("sots_indoor_ssim")
        print(
            f"  {r['variant']:<28} params={r['params']:>10,}  "
            f"SOTS-Indoor PSNR={si if si is None else round(si, 2)} "
            f"SSIM={ss if ss is None else round(ss, 4)}"
        )


if __name__ == "__main__":
    main()
