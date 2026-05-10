import pytest


@pytest.mark.skip(reason="Requires trained checkpoints and ONNX exports.")
def test_onnx_parity_placeholder() -> None:
    assert True
