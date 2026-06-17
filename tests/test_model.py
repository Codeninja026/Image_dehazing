"""Architecture tests: parameter count, I/O shape, output range, layer presence."""
import numpy as np
import pytest

from eca_ldnet.config import IMG_SIZE

# The published model has exactly 1,481,871 parameters (1.482M).
EXPECTED_PARAMS = 1_481_871


def test_param_count(model):
    assert model.count_params() == EXPECTED_PARAMS


def test_io_shape(model):
    assert model.input_shape == (None, IMG_SIZE, IMG_SIZE, 3)
    assert model.output_shape == (None, IMG_SIZE, IMG_SIZE, 3)


def test_forward_pass_range(model):
    x = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype("float32")
    y = model.predict(x, verbose=0)
    assert y.shape == (1, IMG_SIZE, IMG_SIZE, 3)
    assert np.isfinite(y).all()
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_dual_attention_and_physics_present(model):
    names = [type(layer).__name__ for layer in model.layers]
    assert "ECABlock" in names
    assert "PixelAttention" in names
    assert "PhysicsCorrectionLayer" in names


def test_physics_module_outputs(model):
    layer_names = [layer.name for layer in model.layers]
    assert "atm_light" in layer_names      # estimated atmospheric light A
    assert "transmission" in layer_names   # estimated transmission map t


def test_roundtrip_save_load(model, tmp_path):
    import tensorflow as tf

    from eca_ldnet.layers import CUSTOM_OBJECTS

    path = tmp_path / "rt.keras"
    model.save(path)
    reloaded = tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)
    assert reloaded.count_params() == EXPECTED_PARAMS
