# ECA-LDNet: Efficient Channel-Attention Lightweight Dehazing Network

A **1.48M-parameter**, physics-guided, dual-attention network for single-image
dehazing. ECA-LDNet pairs **Efficient Channel Attention (ECA)** with
**Pixel (spatial) Attention** inside a depthwise-separable U-Net, and embeds an
**Atmospheric-Scattering-Model (ASM) physics-correction layer** that estimates
atmospheric light `A` and a transmission map `t` to refine the output. It reaches
**32.53 dB / 0.972 SSIM on SOTS-Indoor** with ~13 FPS inference — competitive with
models 17× larger.

> This repository is a cleaned, installable, and tested refactor of the original
> research notebooks. The model code now lives in the `eca_ldnet/` package, with
> the original notebooks preserved under `notebooks/` for full reproducibility.

---

## Highlights

- **Lightweight:** 1,481,871 parameters (5.65 MB), depthwise-separable convolutions throughout.
- **Dual attention:** ECA (channel) + Pixel Attention (spatial) in every residual block.
- **Physics-guided:** transmission-derived haze-density gating in the decoder and an
  ASM correction head (`J = (I − A)/(t + ε) + A`) blended into the output.
- **Real-time:** ~72–82 ms / 256×256 image (≈12–14 FPS) on a single GPU.
- **Reproducible:** multi-stage progressive training (Charbonnier + SSIM + edge + FFT + VGG perceptual).

## Benchmark Results

| Method | Year | Attention | PSNR (SOTS-Indoor) | SSIM | Params |
|---|---|---|---|---|---|
| DCP | 2009 | None | 16.62 | 0.818 | — |
| AOD-Net | 2017 | None | 19.06 | 0.850 | 0.002M |
| GridDehazeNet | 2019 | Attention | 32.16 | 0.984 | 0.956M |
| FFA-Net | 2020 | CBAM | 36.39 | 0.989 | 4.456M |
| DehazeFormer-B | 2023 | Coord. Attn | 33.26 | 0.983 | 25.44M |
| **ECA-LDNet (Ours)** | **2026** | **ECA + Pixel** | **32.53** | **0.972** | **1.482M** |

Full results across benchmarks (released `stage5` model):

| Dataset | PSNR (dB) | SSIM |
|---|---|---|
| SOTS-Indoor | 32.53 | 0.9717 |
| SOTS-Outdoor | 31.94 | 0.9769 |
| RESIDE-6K | 30.20 | 0.9641 |

See [`results/final_evaluation/final_report.txt`](results/final_evaluation/final_report.txt)
and [`results/model_architecture_summary.txt`](results/model_architecture_summary.txt).

## Architecture

```
Hazy (256×256×3)
   │
   ├─ Encoder: 4× [Residual(DWSConv×2 → ECA → PixelAttention) → MaxPool]  (32→64→128→256)
   │
   ├─ Bottleneck (512) ──► Physics head:  A (atm. light, 3)   t (transmission, 16×16×1)
   │
   ├─ Decoder: 4× [Up → Concat skip → Residual], gated by (1 − t) haze-density maps
   │
   └─ Refine → raw sigmoid (256×256×3)
              └─ PhysicsCorrectionLayer(raw, A, t) ──► Dehazed (256×256×3)
```

## Installation

```bash
git clone https://github.com/manpreet28111995/Image-Dehazing-using-ECA-LDNet.git
cd Image-Dehazing-using-ECA-LDNet
pip install -e .          # or: pip install -r requirements.txt
```

Python ≥ 3.9, TensorFlow 2.16–2.19, **Keras ≥ 3.9** (required to load the released
`.keras` checkpoints — earlier Keras 3 releases reject their serialized
`quantization_config` field).

## Quick Start

**Dehaze a single image** with a released checkpoint:

```bash
python -m eca_ldnet.inference \
    --model checkpoints/stage5_final_ema_model.keras \
    --input path/to/hazy.png \
    --output dehazed.png
```

**Build the model in Python:**

```python
from eca_ldnet import build_eca_ldnet
model = build_eca_ldnet()
model.summary()                 # 1,481,871 params
```

**Evaluate on benchmarks:**

```bash
python -m eca_ldnet.evaluate \
    --model checkpoints/stage5_final_ema_model.keras \
    --sots /path/to/SOTS --reside6k /path/to/RESIDE-6K
```

**Train from scratch (3-stage schedule):**

```bash
python -m eca_ldnet.train --reside6k /data/RESIDE-6K --its /data/ITS --out ./runs
```

## Datasets

| Name | Use | Source |
|---|---|---|
| RESIDE-6K | train + test | `kmljts/reside-6k` |
| ITS (indoor) | train | `balraj98/indoor-training-set-its-residestandard` |
| SOTS (indoor/outdoor) | test | `balraj98/synthetic-objective-testing-set-sots-reside` |

## Repository Layout

```
eca_ldnet/                  Installable package
  layers.py                 ECABlock, PixelAttention, PhysicsCorrectionLayer
  blocks.py                 DWS conv, residual, encoder/decoder blocks
  model.py                  build_eca_ldnet() + load_pretrained()
  losses.py                 Charbonnier/SSIM/edge/FFT/VGG losses + metrics
  data.py                   Dataset discovery, preprocessing, tf.data pipeline
  callbacks.py              Warmup-cosine LR schedule + logging callbacks
  train.py                  Multi-stage training CLI
  evaluate.py               Benchmark PSNR/SSIM + latency/MACs CLI
  inference.py              Single-image dehazing CLI
tests/                      pytest suite (param count, shapes, losses, round-trip)
notebooks/                  Original Kaggle notebooks (train / upgrade / test)
checkpoints/                Released stage-5 models (.keras)
results/                    Per-stage histories, plots, final evaluation report
docs/                       Project report
```

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

Tests verify the exact parameter count (1,481,871), I/O shapes, output range,
presence of the dual-attention and physics modules, save/load round-trip, and
loss/metric correctness. They skip automatically if TensorFlow is not installed.

## Citation

```bibtex
@software{reddy2026ecaldnet,
  author = {Manpreet Singh, Sai Deekshith Lekkalla, Rohith Reddy Bellibatlu, Manmeet Singh Kapoor},
  title  = {ECA-LDNet: Efficient Channel-Attention Lightweight Dehazing Network},
  year   = {2026},
  url    = {https://github.com/manpreet28111995/Image-Dehazing-using-ECA-LDNet}
}
```

## License

Released under the [MIT License](LICENSE).
