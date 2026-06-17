"""Central configuration for ECA-LDNet.

These values mirror the constants used in the original training notebook
(``notebooks/01_train_full_pipeline.ipynb``) so that the refactored package
reproduces the published 1.482M-parameter model exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# --- Data -----------------------------------------------------------------
IMG_SIZE = 256
CHANNELS = 3
SEED = 42

# --- Architecture ---------------------------------------------------------
ENC_FILTERS: List[int] = [32, 64, 128, 256, 512]
DEC_FILTERS: List[int] = [256, 128, 64, 32]
ECA_KERNEL = 3

# Physics correction layer
PHYS_EPS = 0.1
PHYS_BLEND = 0.08

# --- Regularization (AdamW handles weight decay, so no explicit L2) --------
SPATIAL_DROP = 0.05
BOTTLENECK_DROP = 0.10

# --- Training -------------------------------------------------------------
BATCH_SIZE = 16
BATCH_SIZE_S3 = 8

# Multi-stage schedule (110 epochs total for the core 3-stage pipeline)
LR_S1, EPOCHS_S1, WARMUP_S1 = 1e-3, 60, 5
LR_S2, EPOCHS_S2, WARMUP_S2 = 1e-4, 30, 2
LR_S3, EPOCHS_S3 = 5e-5, 20

PATIENCE_ES = 15
PATIENCE_LR = 7
MIN_LR = 1e-7

# Combined-loss weights (Stage 1 / Stage 2)
W_MAE, W_SSIM, W_EDGE, W_FREQ, W_PERCEP = 0.35, 0.20, 0.10, 0.10, 0.25

# Augmentation
AUG_BRIGHTNESS = 0.08
AUG_NOISE_STD = 0.01
AUG_CONTRAST = 0.10


@dataclass
class TrainConfig:
    """Bundles the hyper-parameters for one training stage."""

    stage: int
    lr: float
    epochs: int
    warmup_epochs: int = 0
    batch_size: int = BATCH_SIZE
    img_size: int = IMG_SIZE
    enc_filters: List[int] = field(default_factory=lambda: list(ENC_FILTERS))
    dec_filters: List[int] = field(default_factory=lambda: list(DEC_FILTERS))


STAGE1 = TrainConfig(stage=1, lr=LR_S1, epochs=EPOCHS_S1, warmup_epochs=WARMUP_S1)
STAGE2 = TrainConfig(stage=2, lr=LR_S2, epochs=EPOCHS_S2, warmup_epochs=WARMUP_S2)
STAGE3 = TrainConfig(stage=3, lr=LR_S3, epochs=EPOCHS_S3, batch_size=BATCH_SIZE_S3)
