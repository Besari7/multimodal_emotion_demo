from pathlib import Path


def test_project_scaffold_exists() -> None:
    assert Path("src/cli/batch_infer.py").exists()
    assert Path("src/serving/export_onnx.py").exists()
