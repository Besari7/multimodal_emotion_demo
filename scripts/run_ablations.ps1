Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

param(
    [ValidateSet("oracle", "asr")]
    [string]$Regime = "asr"
)

python -m src.cli.evaluate --suite full --regime $Regime
