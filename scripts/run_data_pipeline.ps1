Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

python -m src.cli.prepare_data --regime oracle
python -m src.cli.prepare_data --regime asr --run-asr
