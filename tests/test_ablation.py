"""Ablation-variant tests: each knock-out builds and changes params as expected."""
import pytest

tf = pytest.importorskip("tensorflow")

from eca_ldnet.model import build_eca_ldnet  # noqa: E402

FULL = 1_481_871


def test_full_default_unchanged():
    assert build_eca_ldnet(name="t").count_params() == FULL


def test_eca_branch_is_essentially_free():
    # ECA is a single bias-free k=3 Conv1D per unit -> 3 params each, 9 units.
    delta = FULL - build_eca_ldnet(name="t", use_eca=False).count_params()
    assert delta == 27


def test_pixel_attention_cost():
    delta = FULL - build_eca_ldnet(name="t", use_pa=False).count_params()
    assert delta == 109_433


def test_physics_module_cost():
    delta = FULL - build_eca_ldnet(name="t", use_physics=False).count_params()
    assert delta == 51_588


def test_ablation_variants_forward_pass():
    import numpy as np
    x = np.random.rand(1, 256, 256, 3).astype("float32")
    for kw in (dict(use_eca=False), dict(use_pa=False), dict(use_physics=False)):
        y = build_eca_ldnet(name="t", **kw).predict(x, verbose=0)
        assert y.shape == (1, 256, 256, 3)
        assert np.isfinite(y).all()
