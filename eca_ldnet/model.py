"""Full ECA-LDNet architecture assembly.

A U-Net-style encoder-decoder (depths 32/64/128/256, bottleneck 512) with dual
attention in every residual block, an embedded physics-guidance module that
estimates atmospheric light ``A`` and a transmission map ``t`` from the
bottleneck, transmission-derived haze-density gating in the decoder, and a final
:class:`~eca_ldnet.layers.PhysicsCorrectionLayer`. The assembled model has
1,481,871 parameters (1.482M).
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers

from .blocks import decoder_block, dws_conv_block, encoder_block, residual_block
from .config import BOTTLENECK_DROP, CHANNELS, DEC_FILTERS, ENC_FILTERS, IMG_SIZE, PHYS_BLEND, PHYS_EPS
from .layers import PhysicsCorrectionLayer


def build_eca_ldnet(
    img_size: int = IMG_SIZE,
    use_eca: bool = True,
    use_pa: bool = True,
    use_physics: bool = True,
    name: str = "ECA_LDNet",
) -> Model:
    """Build and return the ECA-LDNet Keras functional model.

    The three boolean flags exist for controlled ablation and all default to the
    full configuration (which reproduces the released 1,481,871-parameter model):

    * ``use_eca``     - include the ECA channel-attention branch.
    * ``use_pa``      - include the pixel (spatial) attention branch.
    * ``use_physics`` - include the physics-guidance module (atmospheric light /
      transmission heads, haze-density decoder gating, and the ASM correction
      head). When ``False`` the network emits the raw sigmoid output directly.
    """
    ab = dict(use_eca=use_eca, use_pa=use_pa)
    inputs = layers.Input(shape=(img_size, img_size, CHANNELS), name="hazy_input")

    # ---- ENCODER ----
    x, skip1 = encoder_block(inputs, ENC_FILTERS[0], **ab)   # 256 -> 128
    x, skip2 = encoder_block(x, ENC_FILTERS[1], **ab)        # 128 -> 64
    x, skip3 = encoder_block(x, ENC_FILTERS[2], **ab)        # 64  -> 32
    x, skip4 = encoder_block(x, ENC_FILTERS[3], **ab)        # 32  -> 16

    # ---- BOTTLENECK (16x16x512, no pooling) ----
    x = dws_conv_block(x, ENC_FILTERS[4])
    x = residual_block(x, ENC_FILTERS[4], drop_rate=0, **ab)
    x = layers.Dropout(BOTTLENECK_DROP)(x)
    bottleneck = x

    if use_physics:
        # ---- PHYSICS GUIDANCE MODULE ----
        a = layers.GlobalAveragePooling2D()(bottleneck)
        a = layers.Dense(64, activation="relu")(a)
        a = layers.Dropout(0.1)(a)
        a = layers.Dense(32, activation="relu")(a)
        a = layers.Dense(3, activation="sigmoid", dtype="float32", name="atm_light")(a)

        t = layers.Conv2D(32, 1, padding="same")(bottleneck)
        t = layers.BatchNormalization()(t)
        t = layers.Activation("relu")(t)
        t = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", padding="same", name="transmission")(t)

        t_16 = layers.UpSampling2D(2, interpolation="bilinear")(t)   # 32x32x1
        t_32 = layers.UpSampling2D(4, interpolation="bilinear")(t)   # 64x64x1

    # ---- DECODER (with haze-density gating when physics is enabled) ----
    x = decoder_block(x, skip4, DEC_FILTERS[0], **ab)        # 16 -> 32
    if use_physics:
        x = layers.Multiply(name="phys_d4")([x, 1.0 - t_16])
    x = decoder_block(x, skip3, DEC_FILTERS[1], **ab)        # 32 -> 64
    if use_physics:
        x = layers.Multiply(name="phys_d3")([x, 1.0 - t_32])
    x = decoder_block(x, skip2, DEC_FILTERS[2], **ab)        # 64 -> 128
    x = decoder_block(x, skip1, DEC_FILTERS[3], **ab)        # 128 -> 256

    # ---- FINAL REFINEMENT + OUTPUT ----
    x = dws_conv_block(x, 16)
    raw = layers.Conv2D(3, 1, padding="same", use_bias=False, dtype="float32", name="raw_conv")(x)
    raw = layers.Activation("sigmoid", dtype="float32", name="raw_sigmoid")(raw)

    if use_physics:
        t_full = layers.UpSampling2D(img_size // 16, interpolation="bilinear", name="t_full")(t)
        output = PhysicsCorrectionLayer(
            eps=PHYS_EPS, blend=PHYS_BLEND, dtype="float32", name="physics_output"
        )([raw, a, t_full])
    else:
        output = raw

    return Model(inputs=inputs, outputs=output, name=name)


def load_pretrained(path: str, compile: bool = False) -> Model:
    """Load a released ``.keras`` checkpoint with the required custom objects."""
    from .layers import CUSTOM_OBJECTS

    return tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=compile)
