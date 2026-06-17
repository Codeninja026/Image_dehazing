"""Single-image / batched dehazing inference utilities + CLI.

Example
-------
    python -m eca_ldnet.inference --model checkpoints/stage5_final_ema_model.keras \\
        --input hazy.png --output dehazed.png
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from .config import IMG_SIZE
from .data import preprocess_hazy


def dehaze_array(model, hazy_rgb01: np.ndarray) -> np.ndarray:
    """Run the model on a single HxWx3 float image in [0, 1]; return float image."""
    pred = model.predict(hazy_rgb01[np.newaxis], verbose=0)[0]
    return np.clip(pred, 0, 1).astype(np.float32)


def dehaze_file(model, in_path: str, out_path: str, size: int = IMG_SIZE) -> str:
    """Dehaze an image file and write the result to ``out_path``."""
    img = cv2.imread(in_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {in_path}")
    h0, w0 = img.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(img, (size, size)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = preprocess_hazy(rgb)
    pred = dehaze_array(model, rgb)
    pred = cv2.resize(pred, (w0, h0))
    bgr = cv2.cvtColor((pred * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, bgr)
    return out_path


def main():
    p = argparse.ArgumentParser(description="ECA-LDNet single-image dehazing")
    p.add_argument("--model", required=True, help="Path to .keras checkpoint")
    p.add_argument("--input", required=True, help="Hazy input image")
    p.add_argument("--output", default="dehazed.png", help="Output path")
    args = p.parse_args()

    from .model import load_pretrained

    model = load_pretrained(args.model)
    out = dehaze_file(model, args.input, args.output)
    print(f"Saved dehazed image to: {out}")


if __name__ == "__main__":
    main()
