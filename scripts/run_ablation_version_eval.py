#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECA-LDNet evaluation-only ablation study.

This script does NOT train anything.
It loads the 5 already-trained checkpoints and evaluates them on:
- SOTS Indoor
- SOTS Outdoor
- RESIDE-6K Test

Outputs:
- /kaggle/working/ablation_out/ablation_results.csv
- /kaggle/working/ablation_out/ablation_results.md
- printed summary table in the notebook output

Edit only the CONFIG block below.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eca_ldnet import config as C  # noqa: E402
from eca_ldnet.data import (  # noqa: E402
    find_dir,
    get_gt_path,
    load_image_pair,
    set_global_seed,
)
from eca_ldnet.metrics import lpips_available, lpips_distance, ms_ssim, psnr, ssim  # noqa: E402
from eca_ldnet.model import build_eca_ldnet  # noqa: E402


# ============================================================
# CONFIG: EDIT ONLY THIS BLOCK
# ============================================================
RESIDE6K_BASE = "/kaggle/input/datasets/kmljts/reside-6k/RESIDE-6K"
ITS_BASE = "/kaggle/input/datasets/balraj98/indoor-training-set-its-residestandard"
SOTS_BASE = "/kaggle/input/datasets/balraj98/synthetic-objective-testing-set-sots-reside"

# Folder that contains the 5 model files from your screenshot
MODELS_ROOT = "/kaggle/input/models/codeninjalucky/all-ablation-models/tensorflow2/default/1"

OUT_DIR = "/kaggle/working/ablation_out"

MODEL_SPECS = [
    {
        "variant": "Full (ECA+PA+Physics)",
        "filename": "Full_(ECAPAPhysics)_epoch_28.keras",
        "flags": dict(),
    },
    {
        "variant": "w/o ECA",
        "filename": "w_o_ECA_epoch_28.keras",
        "flags": dict(use_eca=False),
    },
    {
        "variant": "w/o Pixel Attention",
        "filename": "w_o_Pixel_Attention_epoch_28.keras",
        "flags": dict(use_pa=False),
    },
    {
        "variant": "w/o Physics guidance",
        "filename": "w_o_Physics_guidance_epoch_28.keras",
        "flags": dict(use_physics=False),
    },
    {
        "variant": "w/o all attention",
        "filename": "w_o_all_attention_epoch_28.keras",
        "flags": dict(use_eca=False, use_pa=False),
    },
]
# ============================================================


def _normalize_name(name: str) -> str:
    return name.lower().lstrip("._")


def _resolve_model_path(root: str, filename: str) -> str | None:
    """
    Resolve the checkpoint path.
    Handles:
    - exact filename
    - leading '._' macOS-style hidden file prefix
    - recursive search inside MODELS_ROOT
    """
    if not root:
        return None

    root_path = Path(root)
    if not root_path.exists():
        return None

    exact = root_path / filename
    if exact.exists():
        return str(exact)

    # Handle macOS hidden file prefix
    prefixed = root_path / f"._{filename}"
    if prefixed.exists():
        return str(prefixed)

    # Fallback: recursive search by normalized name
    target = _normalize_name(filename)
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if _normalize_name(fn) == target:
                return os.path.join(dirpath, fn)

    return None


def _gather_train(reside6k: str | None, its: str | None):
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


def collect_pairs(hazy_dir, gt_dir, dataset_name="dataset"):
    """
    Collect aligned hazy/GT file pairs from a folder.
    """
    pairs = []
    if not hazy_dir or not gt_dir:
        return pairs

    files = sorted(
        f for f in os.listdir(hazy_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    )

    for fname in files:
        gp = get_gt_path(fname, gt_dir)
        if gp:
            pairs.append((os.path.join(hazy_dir, fname), gp))

    print(f"  {dataset_name:<20}: {len(pairs):5d} pairs")
    return pairs


def _eval_dir(model, hazy_dir, gt_dir, limit=None):
    """
    Evaluate a model on one hazy/GT directory pair.
    Returns mean PSNR / SSIM / MS-SSIM / LPIPS.
    """
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


def _load_variant_model(flags, model_path: str):
    """
    Build the exact architecture and load weights from the saved .keras file.
    This avoids deserialization issues and worked for your checkpoints.
    """
    model = build_eca_ldnet(name="eval", **flags)
    model.load_weights(model_path)
    return model


def main():
    set_global_seed(C.SEED)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Resolve datasets
    r6k_train_hazy = find_dir(RESIDE6K_BASE, ["train/hazy", "Train/hazy"])
    r6k_train_gt = find_dir(RESIDE6K_BASE, ["train/GT", "train/gt", "train/clear"])
    r6k_test_hazy = find_dir(RESIDE6K_BASE, ["test/hazy", "Test/hazy"])
    r6k_test_gt = find_dir(RESIDE6K_BASE, ["test/GT", "test/gt", "test/clear"])

    its_hazy = find_dir(ITS_BASE, ["hazy", "Hazy", "train/hazy"])
    its_gt = find_dir(ITS_BASE, ["clear", "GT", "gt", "train/clear", "train/GT"])

    sots_in_hazy = find_dir(SOTS_BASE, ["indoor/hazy", "SOTS/indoor/hazy"])
    sots_in_gt = find_dir(SOTS_BASE, ["indoor/clear", "indoor/gt", "indoor/GT", "SOTS/indoor/GT"])

    sots_out_hazy = find_dir(SOTS_BASE, ["outdoor/hazy", "SOTS/outdoor/hazy"])
    sots_out_gt = find_dir(SOTS_BASE, ["outdoor/clear", "outdoor/Clear", "outdoor/GT", "SOTS/outdoor/GT"])

    print("\n================ DATASET PATH VERIFICATION ================")
    for name, path in [
        ("R6K Train Hazy", r6k_train_hazy),
        ("R6K Train GT", r6k_train_gt),
        ("R6K Test Hazy", r6k_test_hazy),
        ("R6K Test GT", r6k_test_gt),
        ("ITS Hazy", its_hazy),
        ("ITS GT", its_gt),
        ("SOTS In Hazy", sots_in_hazy),
        ("SOTS In GT", sots_in_gt),
        ("SOTS Out Hazy", sots_out_hazy),
        ("SOTS Out GT", sots_out_gt),
    ]:
        if path:
            print(f"  ✓ {name:<15} {path}")
        else:
            print(f"  ✗ {name:<15} NOT FOUND")
    print("==========================================================\n")

    # Build evaluation sets
    eval_sets = [
        ("SOTS_Indoor", sots_in_hazy, sots_in_gt),
        ("SOTS_Outdoor", sots_out_hazy, sots_out_gt),
        ("RESIDE6K_Test", r6k_test_hazy, r6k_test_gt),
    ]

    rows = []

    for spec in MODEL_SPECS:
        variant = spec["variant"]
        filename = spec["filename"]
        flags = spec["flags"]

        model_path = _resolve_model_path(MODELS_ROOT, filename)
        if model_path is None:
            raise FileNotFoundError(
                f"Could not find checkpoint for '{variant}'. "
                f"Looked for '{filename}' under: {MODELS_ROOT}"
            )

        print(f"\n===== Loading variant: {variant} =====")
        print(f"Model file: {model_path}")

        model = _load_variant_model(flags, model_path)
        nparams = model.count_params()

        row = {
            "variant": variant,
            "model_file": os.path.basename(model_path),
            "params": nparams,
        }

        for set_name, hazy_dir, gt_dir in eval_sets:
            result = _eval_dir(model, hazy_dir, gt_dir)
            if result is None:
                row[f"{set_name}_psnr"] = None
                row[f"{set_name}_ssim"] = None
                row[f"{set_name}_ms_ssim"] = None
                row[f"{set_name}_lpips"] = None
                row[f"{set_name}_n"] = 0
            else:
                row[f"{set_name}_psnr"] = result["psnr"]
                row[f"{set_name}_ssim"] = result["ssim"]
                row[f"{set_name}_ms_ssim"] = result["ms_ssim"]
                row[f"{set_name}_lpips"] = result.get("lpips")
                row[f"{set_name}_n"] = result["n"]

        rows.append(row)

        print(
            f"  {variant:<26} "
            f"SOTS-Indoor PSNR={row['SOTS_Indoor_psnr']:.4f} "
            f"SSIM={row['SOTS_Indoor_ssim']:.4f}"
        )

    df = pd.DataFrame(rows)

    # Pretty table display
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n================ FINAL ABLATION TABLE ================\n")
    print(df.to_string(index=False))

    csv_path = os.path.join(OUT_DIR, "ablation_results.csv")
    md_path = os.path.join(OUT_DIR, "ablation_results.md")

    df.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved MD : {md_path}")
    print(f"Output folder: {OUT_DIR}")

    # Optional: small summary
    print("\n================ SUMMARY ================")
    for _, r in df.iterrows():
        print(
            f"{r['variant']:<26} | "
            f"SOTS-Indoor PSNR={r['SOTS_Indoor_psnr']:.4f} SSIM={r['SOTS_Indoor_ssim']:.4f} | "
            f"SOTS-Outdoor PSNR={r['SOTS_Outdoor_psnr']:.4f} SSIM={r['SOTS_Outdoor_ssim']:.4f} | "
            f"RESIDE-6K PSNR={r['RESIDE6K_Test_psnr']:.4f} SSIM={r['RESIDE6K_Test_ssim']:.4f}"
        )


if __name__ == "__main__":
    main()
