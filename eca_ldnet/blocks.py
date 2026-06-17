"""Composite building blocks: depthwise-separable conv, residual, enc/dec.

The attention flags (``use_eca`` / ``use_pa``) make the dual-attention modules
individually switchable so that controlled ablation variants can be built from
exactly the same code path as the full model.
"""
from __future__ import annotations

from tensorflow.keras import layers

from .config import ECA_KERNEL, SPATIAL_DROP
from .layers import ECABlock, PixelAttention


def dws_conv_block(x, filters, kernel_size=3, strides=1):
    """Depthwise-separable conv block: DWConv -> BN -> ReLU6 -> 1x1 -> BN -> ReLU6."""
    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)
    return x


def residual_block(x, filters, use_eca=True, use_pa=True, drop_rate=SPATIAL_DROP):
    """Two DWS conv blocks + optional dual attention (ECA then PixelAttention) + residual.

    ``use_eca`` and ``use_pa`` toggle the channel- and spatial-attention branches
    independently, which is what the ablation driver flips.
    """
    skip = x
    if x.shape[-1] != filters:
        skip = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
        skip = layers.BatchNormalization()(skip)
    x = dws_conv_block(x, filters)
    x = dws_conv_block(x, filters)
    if use_eca:
        x = ECABlock(kernel_size=ECA_KERNEL)(x)
    if use_pa:
        x = PixelAttention()(x)
    if drop_rate > 0:
        x = layers.SpatialDropout2D(drop_rate)(x)
    x = layers.Add()([x, skip])
    return x


def encoder_block(x, filters, use_eca=True, use_pa=True):
    """Residual block (kept as skip) followed by 2x max-pool downsample."""
    x = residual_block(x, filters, use_eca=use_eca, use_pa=use_pa, drop_rate=SPATIAL_DROP)
    skip = x
    x = layers.MaxPooling2D(2)(x)
    return x, skip


def decoder_block(x, skip, filters, use_eca=True, use_pa=True):
    """Bilinear upsample, concat skip, fuse with 1x1 conv, then a residual block."""
    x = layers.UpSampling2D(2, interpolation="bilinear")(x)
    x = layers.Concatenate()([x, skip])
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)
    x = residual_block(x, filters, use_eca=use_eca, use_pa=use_pa, drop_rate=SPATIAL_DROP / 2)
    return x
