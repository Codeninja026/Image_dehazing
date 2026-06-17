"""Learning-rate schedule and training callbacks."""
from __future__ import annotations

import math

import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks

from .config import MIN_LR


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay to ``min_lr``."""

    def __init__(self, peak_lr, warmup_steps, total_steps, min_lr=MIN_LR):
        self.peak_lr = float(peak_lr)
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.peak_lr * (step / (self.warmup_steps + 1e-8))
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps + 1e-8)
        cos_lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * tf.clip_by_value(progress, 0.0, 1.0))
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cos_lr)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }


class CSVMetricsLogger(callbacks.Callback):
    """Append per-epoch metrics (incl. current LR) to a CSV file."""

    def __init__(self, path, stage):
        super().__init__()
        self.path = path
        self.stage = stage
        self.records = []

    def on_epoch_end(self, epoch, logs=None):
        row = {"epoch": epoch, "stage": self.stage}
        row.update(logs or {})
        try:
            opt = self.model.optimizer
            lr = opt.learning_rate
            row["lr"] = float(lr(opt.iterations) if callable(lr) else lr)
        except Exception:
            row["lr"] = 0.0
        self.records.append(row)
        pd.DataFrame(self.records).to_csv(self.path, index=False)


def make_callbacks(stage, model_dir, history_dir, monitor="val_psnr_metric", patience=15):
    """Standard checkpoint + early-stopping + CSV-logging callback set."""
    return [
        callbacks.ModelCheckpoint(
            f"{model_dir}/stage{stage}_best.keras",
            monitor=monitor,
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        CSVMetricsLogger(f"{history_dir}/stage{stage}_history.csv", stage),
    ]
