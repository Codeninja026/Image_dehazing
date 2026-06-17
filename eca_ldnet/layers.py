"""Custom Keras layers for ECA-LDNet.

Three building blocks define the network's identity:

* :class:`ECABlock`            - Efficient Channel Attention (1-D conv over the
  channel descriptor; parameter-free up to a tiny ``k``-tap kernel).
* :class:`PixelAttention`      - lightweight spatial attention that learns
  *where* in the feature map to focus.
* :class:`PhysicsCorrectionLayer` - blends the network output with an
  atmospheric-scattering-model (ASM) reconstruction ``J = (I - A) / t + A``.

These definitions are byte-for-byte compatible with the layers serialised into
the released ``.keras`` checkpoints, so saved models load with
``custom_objects=CUSTOM_OBJECTS``.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


class ECABlock(layers.Layer):
    """Efficient Channel Attention (Wang et al., CVPR 2020).

    Squeeze with global average pooling, model cross-channel interaction with a
    single 1-D convolution of size ``kernel_size``, and gate the input with the
    resulting per-channel weights.
    """

    def __init__(self, kernel_size: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.channels = int(input_shape[-1])
        self.conv = layers.Conv1D(1, self.kernel_size, padding="same", use_bias=False)
        self.gap = layers.GlobalAveragePooling2D()
        super().build(input_shape)

    def call(self, x):
        b = tf.shape(x)[0]
        gap = self.gap(x)                                  # (B, C)
        gap = tf.reshape(gap, [b, self.channels, 1])       # (B, C, 1)
        attn = tf.sigmoid(self.conv(gap))                  # (B, C, 1)
        attn = tf.reshape(attn, [b, 1, 1, self.channels])  # (B, 1, 1, C)
        return x * attn

    def get_config(self):
        cfg = super().get_config()
        cfg["kernel_size"] = self.kernel_size
        return cfg


class PixelAttention(layers.Layer):
    """Spatial (pixel-wise) attention.

    A 1x1 bottleneck conv (C -> C/4) followed by a 1x1 conv to a single-channel
    sigmoid map that re-weights every spatial location.
    """

    def build(self, input_shape):
        c = int(input_shape[-1])
        self.conv1 = layers.Conv2D(max(8, c // 4), 1, padding="same", use_bias=False)
        self.conv2 = layers.Conv2D(1, 1, padding="same", activation="sigmoid")
        super().build(input_shape)

    def call(self, x):
        attn = tf.nn.relu(self.conv1(x))
        attn = self.conv2(attn)            # (B, H, W, 1)
        return x * attn

    def get_config(self):
        return super().get_config()


class PhysicsCorrectionLayer(layers.Layer):
    """Atmospheric-scattering-model guided refinement.

    Given the raw network output ``img``, an estimated atmospheric light ``A``
    (B, 3) and a transmission map ``t`` (B, H, W, 1), reconstruct the scene
    radiance ``J = (I - A) / (t + eps) + A`` and blend it lightly into the
    output so the physics acts as a soft prior rather than a hard constraint.
    """

    def __init__(self, eps: float = 0.1, blend: float = 0.08, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.blend = blend

    def call(self, inputs):
        img, a, t = inputs
        a_bc = tf.reshape(a, [-1, 1, 1, 3])
        a_bc = tf.broadcast_to(a_bc, tf.shape(img))
        t_bc = tf.broadcast_to(t, tf.shape(img))
        j = (img - a_bc) / (t_bc + self.eps) + a_bc
        j = tf.clip_by_value(j, 0.0, 1.0)
        return img * (1.0 - self.blend) + j * self.blend

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"eps": self.eps, "blend": self.blend})
        return cfg


# Used when loading released checkpoints with tf.keras.models.load_model(...)
CUSTOM_OBJECTS = {
    "ECABlock": ECABlock,
    "PixelAttention": PixelAttention,
    "PhysicsCorrectionLayer": PhysicsCorrectionLayer,
}
