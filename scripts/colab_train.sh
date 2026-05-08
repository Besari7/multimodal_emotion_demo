#!/usr/bin/env bash
set -euo pipefail

REGIME=${1:-asr}

python -m pip install --upgrade pip
python -m pip install -r requirements-colab.txt

python -m src.cli.prepare_data --regime oracle
python -m src.cli.prepare_data --regime asr --run-asr

for FOLD in 0 1 2 3 4; do
  python -m src.cli.train_unimodal --branch audio --regime ${REGIME} --fold ${FOLD}
  python -m src.cli.train_unimodal --branch video --regime ${REGIME} --fold ${FOLD}
  python -m src.cli.train_unimodal --branch text --regime ${REGIME} --fold ${FOLD}
  python -m src.cli.train_fusion --mode late --regime ${REGIME} --fold ${FOLD}
done

python -m src.cli.evaluate --suite full --regime ${REGIME}
python -m src.serving.export_onnx --all
