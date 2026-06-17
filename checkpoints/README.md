# Released Checkpoints

| File | Description |
|---|---|
| `stage5_best.keras` | Best stage-5 model by validation PSNR. |
| `stage5_final_ema_model.keras` | Stage-5 EMA-averaged weights (recommended for inference). |

Per-stage checkpoints and training histories for stages 1–4 are kept under
[`../results/each_stage_results/`](../results/each_stage_results/).

## Loading

```python
from eca_ldnet.model import load_pretrained
model = load_pretrained("checkpoints/stage5_final_ema_model.keras")
```

`load_pretrained` supplies the required `custom_objects`
(`ECABlock`, `PixelAttention`, `PhysicsCorrectionLayer`).

> **Note:** these checkpoints were serialized with a recent Keras and require
> **Keras ≥ 3.9** to deserialize. With the Keras 3.0.x that ships by default with
> TensorFlow 2.16, loading fails with
> `Unrecognized keyword arguments passed to Dense: {'quantization_config': None}`.
> Run `pip install "keras>=3.9"` (already pinned in `requirements.txt`).
