"""Data loading, preprocessing, and ``tf.data`` pipeline for ECA-LDNet.

Supports the RESIDE-6K, ITS (indoor), and SOTS (indoor/outdoor) layouts used in
the paper. Hazy inputs receive a mild gamma (0.9) and per-channel min-max
stretch to stabilise contrast before the network.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

from .config import (
    AUG_BRIGHTNESS,
    AUG_CONTRAST,
    AUG_NOISE_STD,
    BATCH_SIZE,
    IMG_SIZE,
    SEED,
)

PairList = List[Tuple[str, str]]


def find_dir(base: Optional[str], candidates: List[str]) -> Optional[str]:
    """Return the first existing, non-empty sub-directory matching a candidate."""
    if not base or not os.path.isdir(base):
        return None
    for c in candidates:
        p = os.path.join(base, c)
        if os.path.isdir(p) and len(os.listdir(p)) > 0:
            return p
    for sub in os.listdir(base):
        sp = os.path.join(base, sub)
        if os.path.isdir(sp):
            for c in candidates:
                p = os.path.join(sp, c)
                if os.path.isdir(p) and len(os.listdir(p)) > 0:
                    return p
    return None


def preprocess_hazy(img: np.ndarray) -> np.ndarray:
    """Gamma 0.9 + per-channel min-max stretch, clipped to [0, 1]."""
    img = np.power(np.clip(img, 0, 1), 0.9).astype(np.float32)
    for c in range(3):
        lo, hi = img[:, :, c].min(), img[:, :, c].max()
        if hi > lo + 1e-6:
            img[:, :, c] = (img[:, :, c] - lo) / (hi - lo)
    return np.clip(img, 0, 1).astype(np.float32)


def load_image_pair(hazy_path: str, gt_path: str, size: int = IMG_SIZE):
    """Read, resize, RGB-convert, and preprocess a (hazy, gt) image pair."""
    hazy = cv2.imread(str(hazy_path))
    gt = cv2.imread(str(gt_path))
    if hazy is None or gt is None:
        return None, None
    hazy = cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    hazy = cv2.resize(hazy, (size, size)).astype(np.float32) / 255.0
    gt = cv2.resize(gt, (size, size)).astype(np.float32) / 255.0
    hazy = preprocess_hazy(hazy)
    return hazy, gt


def get_gt_path(hazy_name: str, gt_dir: str) -> Optional[str]:
    """Resolve the ground-truth path for a hazy filename across naming schemes."""
    stem = Path(hazy_name).stem
    candidates = [
        stem,
        stem.split("_")[0],
        stem.replace("_hazy", ""),
        stem.replace("_fog", ""),
        stem.replace("_synthetic", ""),
    ]
    for cand in candidates:
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".bmp"]:
            p = os.path.join(gt_dir, cand + ext)
            if os.path.exists(p):
                return p
    return None


def collect_pairs(hazy_dir: Optional[str], gt_dir: Optional[str], tag: str = "", limit: Optional[int] = None) -> PairList:
    """Collect matched (hazy, gt) path pairs from a directory."""
    if not hazy_dir or not gt_dir:
        return []
    files = sorted(os.listdir(hazy_dir))
    if limit:
        files = files[:limit]
    pairs: PairList = []
    for hf in files:
        if not hf.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        gp = get_gt_path(hf, gt_dir)
        if gp:
            pairs.append((os.path.join(hazy_dir, hf), gp))
    print(f"  {tag:<22}: {len(pairs):>5} pairs  ({len(files) - len(pairs)} skipped)")
    return pairs


def _load_pair_tf(hazy_path, gt_path):
    def _py_load(hp, gp):
        h, g = load_image_pair(hp.numpy().decode("utf-8"), gp.numpy().decode("utf-8"))
        if h is None:
            h = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.float32)
            g = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.float32)
        return h, g

    hazy, gt = tf.py_function(_py_load, [hazy_path, gt_path], [tf.float32, tf.float32])
    hazy.set_shape([IMG_SIZE, IMG_SIZE, 3])
    gt.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return hazy, gt


def _augment_pair(hazy, gt):
    do_h = tf.random.uniform(()) > 0.5
    hazy = tf.cond(do_h, lambda: tf.image.flip_left_right(hazy), lambda: hazy)
    gt = tf.cond(do_h, lambda: tf.image.flip_left_right(gt), lambda: gt)
    do_v = tf.random.uniform(()) > 0.5
    hazy = tf.cond(do_v, lambda: tf.image.flip_up_down(hazy), lambda: hazy)
    gt = tf.cond(do_v, lambda: tf.image.flip_up_down(gt), lambda: gt)
    k = tf.random.uniform((), 0, 4, dtype=tf.int32)
    hazy = tf.image.rot90(hazy, k)
    gt = tf.image.rot90(gt, k)
    hazy = tf.image.random_brightness(hazy, max_delta=AUG_BRIGHTNESS)
    hazy = tf.image.random_contrast(hazy, 1 - AUG_CONTRAST, 1 + AUG_CONTRAST)
    hazy = hazy + tf.random.normal(tf.shape(hazy), stddev=AUG_NOISE_STD)
    hazy = tf.clip_by_value(hazy, 0.0, 1.0)
    gt = tf.clip_by_value(gt, 0.0, 1.0)
    return hazy, gt


def build_dataset(hazy_list, gt_list, augment=True, batch=BATCH_SIZE, shuffle=True, cache_ds=False):
    """Assemble a prefetching ``tf.data.Dataset`` of (hazy, gt) batches."""
    ds = tf.data.Dataset.from_tensor_slices((hazy_list, gt_list))
    if shuffle:
        ds = ds.shuffle(min(len(hazy_list), 3000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load_pair_tf, num_parallel_calls=tf.data.AUTOTUNE)
    if cache_ds:
        ds = ds.cache()
    if augment:
        ds = ds.map(_augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch, drop_remainder=False)
    return ds.prefetch(tf.data.AUTOTUNE)


def set_global_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
