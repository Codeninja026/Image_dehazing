"""Loss / metric sanity tests (no network weights downloaded)."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from eca_ldnet.losses import (  # noqa: E402
    charbonnier_loss,
    edge_loss,
    frequency_loss,
    mae_loss,
    psnr_metric,
    ssim_loss,
    ssim_metric,
)


@pytest.fixture
def pair():
    rng = np.random.default_rng(0)
    a = rng.random((2, 64, 64, 3)).astype("float32")
    return tf.constant(a), tf.constant(a.copy())


def test_identity_losses_are_zeroish(pair):
    y, yhat = pair
    assert float(mae_loss(y, yhat)) < 1e-6
    assert float(ssim_loss(y, yhat)) < 1e-4
    assert float(edge_loss(y, yhat)) < 1e-4
    assert float(frequency_loss(y, yhat)) < 1e-3
    # Charbonnier has an eps floor, so it is small but non-zero.
    assert float(charbonnier_loss(y, yhat)) < 2e-3


def test_metrics_on_identity(pair):
    y, yhat = pair
    assert float(tf.reduce_mean(ssim_metric(y, yhat))) > 0.99
    assert float(tf.reduce_mean(psnr_metric(y, yhat))) > 60.0


def test_losses_increase_with_corruption(pair):
    y, _ = pair
    noisy = tf.clip_by_value(y + tf.random.normal(tf.shape(y), stddev=0.3), 0.0, 1.0)
    assert float(mae_loss(y, noisy)) > float(mae_loss(y, y))
    assert float(ssim_loss(y, noisy)) > float(ssim_loss(y, y))
