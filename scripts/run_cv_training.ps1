Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

param(
    [ValidateSet("oracle", "asr")]
    [string]$Regime = "asr"
)

for ($fold = 0; $fold -lt 5; $fold++) {
    python -m src.cli.train_unimodal --branch audio --regime $Regime --fold $fold
    python -m src.cli.train_unimodal --branch video --regime $Regime --fold $fold
    python -m src.cli.train_unimodal --branch text --regime $Regime --fold $fold
    python -m src.cli.train_fusion --mode late --regime $Regime --fold $fold
}
