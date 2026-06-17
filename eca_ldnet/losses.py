"""Loss functions and metrics for ECA-LDNet.

The Stage 1/2 objective is a weighted sum of Charbonnier, SSIM, edge (Sobel),
log-FFT frequency, and VGG16 perceptual losses. Stage 3 re-weights toward
Charbonnier + perceptual for final polishing. A VGG16 feature extractor is built
lazily on first use so that importing this module never forces a network
download.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model

from .config import IMG_SIZE, W_EDGE, W_FREQ, W_MAE, W_PERCEP, W_SSIM

_PERCEPTUAL_MODEL = None


def _get_perceptual_model() -> Model:
    """Lazily build a frozen VGG16 feature extractor (adds 0 trainable params)."""
    global _PERCEPTUAL_MODEL
    if _PERCEPTUAL_MODEL is None:
        from tensorflow.keras.applications import VGG16

        base = VGG16(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
        base.trainable = False
        _PERCEPTUAL_MODEL = Model(
            inputs=base.input,
            outputs=[
                base.get_layer("block1_conv2").output,
                base.get_layer("block2_conv2").output,
                base.get_layer("block3_conv3").output,
            ],
            name="vgg_perceptual",
        )
        _PERCEPTUAL_MODEL.trainable = False
    return _PERCEPTUAL_MODEL


def _cast(a, b):
    return tf.cast(a, tf.float32), tf.cast(b, tf.float32)


def mae_loss(y_true, y_pred):
    y_true, y_pred = _cast(y_true, y_pred)
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def charbonnier_loss(y_true, y_pred, eps=1e-3):
    y_true, y_pred = _cast(y_true, y_pred)
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + eps * eps))


def ssim_loss(y_true, y_pred):
    y_true, y_pred = _cast(y_true, y_pred)
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def edge_loss(y_true, y_pred):
    y_true, y_pred = _cast(y_true, y_pred)
    return tf.reduce_mean(tf.abs(tf.image.sobel_edges(y_true) - tf.image.sobel_edges(y_pred)))


def frequency_loss(y_true, y_pred):
    y_true, y_pred = _cast(y_true, y_pred)

    def log_fft(img):
        gray = tf.reduce_mean(img, axis=-1)
        return tf.math.log1p(tf.abs(tf.signal.rfft2d(gray)))

    return tf.reduce_mean(tf.abs(log_fft(y_true) - log_fft(y_pred)))


def perceptual_loss(y_true, y_pred):
    y_true, y_pred = _cast(y_true, y_pred)
    model = _get_perceptual_model()
    mean = tf.constant([123.68, 116.779, 103.939], shape=[1, 1, 1, 3], dtype=tf.float32)
    true_feats = model(y_true * 255.0 - mean, training=False)
    pred_feats = model(y_pred * 255.0 - mean, training=False)
    loss = 0.0
    for t_f, p_f in zip(true_feats, pred_feats):
        loss += tf.reduce_mean(tf.abs(t_f - p_f))
    return loss / 3.0


def combined_loss(y_true, y_pred):
    """Stage 1 / Stage 2 objective."""
    return (
        W_MAE * charbonnier_loss(y_true, y_pred)
        + W_SSIM * ssim_loss(y_true, y_pred)
        + W_EDGE * edge_loss(y_true, y_pred)
        + W_FREQ * frequency_loss(y_true, y_pred)
        + W_PERCEP * perceptual_loss(y_true, y_pred)
    )


def combined_loss_s3(y_true, y_pred):
    """Stage 3 final-polish objective (Charbonnier + perceptual heavy)."""
    return (
        0.40 * charbonnier_loss(y_true, y_pred)
        + 0.20 * ssim_loss(y_true, y_pred)
        + 0.10 * edge_loss(y_true, y_pred)
        + 0.05 * frequency_loss(y_true, y_pred)
        + 0.25 * perceptual_loss(y_true, y_pred)
    )


# ---- Metrics -------------------------------------------------------------
def psnr_metric(y_true, y_pred):
    y_true, y_pred = _cast(y_true, y_pred)
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def ssim_metric(y_true, y_pred):
    y_true, y_pred = _cast(y_true, y_pred)
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


def mae_metric(y_true, y_pred):
    y_true, y_pred = _cast(y_true, y_pred)
    return tf.reduce_mean(tf.abs(y_true - y_pred))
