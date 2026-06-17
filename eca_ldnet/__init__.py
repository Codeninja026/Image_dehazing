"""ECA-LDNet: Efficient Channel Attention Lightweight Dehazing Network.

A 1.48M-parameter, physics-guided, dual-attention (ECA + Pixel Attention)
single-image dehazing network. This package contains a faithful, importable
refactor of the model, losses, data pipeline, and training/evaluation code that
originally lived in the project notebooks.
"""

from .layers import ECABlock, PixelAttention, PhysicsCorrectionLayer, CUSTOM_OBJECTS
from .model import build_eca_ldnet

__version__ = "1.0.0"

__all__ = [
    "ECABlock",
    "PixelAttention",
    "PhysicsCorrectionLayer",
    "CUSTOM_OBJECTS",
    "build_eca_ldnet",
    "__version__",
]
