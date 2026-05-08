Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

python -m src.serving.export_onnx --all
