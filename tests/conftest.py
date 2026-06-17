"""Shared pytest fixtures. Skips the whole suite gracefully if TF is absent."""
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow is required to run the model tests")


@pytest.fixture(scope="session")
def model():
    from eca_ldnet.model import build_eca_ldnet

    return build_eca_ldnet()
