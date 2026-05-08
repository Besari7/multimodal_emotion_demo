# MELD-Only Local Testing

This guide covers a local pipeline test using only the MELD dataset. Do not add RAVDESS, CREMA-D, MOSEI, IEMOCAP, GoEmotions, FER2013, AffectNet, or RAF-DB in this step. Other datasets can be added later.

Target labels:

```text
neutral, joy, sadness, anger, fear, disgust, surprise
```

MELD labels map directly to these labels.

## Local Folder Structure

Place MELD files under the repository without absolute paths:

```text
data/
  raw/
    meld/
      train_sent_emo.csv
      dev_sent_emo.csv
      test_sent_emo.csv
      train_splits/
        dia0_utt0.mp4
      dev_splits_complete/
        dia0_utt0.mp4
      output_repeated_splits_test/
        dia0_utt0.mp4
  processed/
    meld/
  features/
    audio/
      meld/
    video/
      meld/
```

If you already extracted audio clips, keep them in a folder with matching stems such as `dia0_utt0.wav` and pass `--audio-root` to `scripts/prepare_meld_manifest.py`.

## Environment Setup

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Bash:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## GPU Check

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

If CUDA is not available, replace `--device cuda` with `--device cpu` in feature extraction commands.

## Create MELD Manifests

Train:

```bash
python scripts/prepare_meld_manifest.py \
  --csv data/raw/meld/train_sent_emo.csv \
  --video-root data/raw/meld/train_splits \
  --output data/processed/meld/train_raw.jsonl \
  --split train
```

Validation:

```bash
python scripts/prepare_meld_manifest.py \
  --csv data/raw/meld/dev_sent_emo.csv \
  --video-root data/raw/meld/dev_splits_complete \
  --output data/processed/meld/val_raw.jsonl \
  --split val
```

Test:

```bash
python scripts/prepare_meld_manifest.py \
  --csv data/raw/meld/test_sent_emo.csv \
  --video-root data/raw/meld/output_repeated_splits_test \
  --output data/processed/meld/test_raw.jsonl \
  --split test
```

With optional audio roots:

```bash
python scripts/prepare_meld_manifest.py \
  --csv data/raw/meld/train_sent_emo.csv \
  --video-root data/raw/meld/train_splits \
  --audio-root data/raw/meld/train_audio \
  --output data/processed/meld/train_raw.jsonl \
  --split train
```

## Manifest Stats

```bash
python scripts/report_manifest_stats.py \
  --manifest data/processed/meld/train_raw.jsonl \
  --output data/processed/meld/train_raw_stats.json
```

```bash
python scripts/report_manifest_stats.py \
  --manifest data/processed/meld/val_raw.jsonl
```

```bash
python scripts/report_manifest_stats.py \
  --manifest data/processed/meld/test_raw.jsonl
```

## Small Subsets

```bash
python scripts/create_manifest_subset.py \
  --input data/processed/meld/train_raw.jsonl \
  --output data/processed/meld/train_raw_small.jsonl \
  --max-per-label 5 \
  --seed 42
```

```bash
python scripts/create_manifest_subset.py \
  --input data/processed/meld/val_raw.jsonl \
  --output data/processed/meld/val_raw_small.jsonl \
  --max-per-label 3 \
  --seed 42
```

For a test subset:

```bash
python scripts/create_manifest_subset.py \
  --input data/processed/meld/test_raw.jsonl \
  --output data/processed/meld/test_raw_small.jsonl \
  --max-per-label 3 \
  --seed 42
```

## Feature Extraction: Small Subset

Audio:

```bash
PYTHONPATH=src python scripts/extract_audio_features.py \
  --input-manifest data/processed/meld/train_raw_small.jsonl \
  --output-manifest data/processed/meld/train_audio_small.jsonl \
  --feature-dir data/features/audio/meld/train_small \
  --device cuda
```

```bash
PYTHONPATH=src python scripts/extract_audio_features.py \
  --input-manifest data/processed/meld/val_raw_small.jsonl \
  --output-manifest data/processed/meld/val_audio_small.jsonl \
  --feature-dir data/features/audio/meld/val_small \
  --device cuda
```

Video:

```bash
PYTHONPATH=src python scripts/extract_video_features.py \
  --input-manifest data/processed/meld/train_audio_small.jsonl \
  --output-manifest data/processed/meld/train_small.jsonl \
  --feature-dir data/features/video/meld/train_small \
  --num-frames 8 \
  --device cuda
```

```bash
PYTHONPATH=src python scripts/extract_video_features.py \
  --input-manifest data/processed/meld/val_audio_small.jsonl \
  --output-manifest data/processed/meld/val_small.jsonl \
  --feature-dir data/features/video/meld/val_small \
  --num-frames 8 \
  --device cuda
```

## Feature Extraction: Full MELD

Train:

```bash
PYTHONPATH=src python scripts/extract_audio_features.py \
  --input-manifest data/processed/meld/train_raw.jsonl \
  --output-manifest data/processed/meld/train_audio.jsonl \
  --feature-dir data/features/audio/meld/train \
  --device cuda
```

```bash
PYTHONPATH=src python scripts/extract_video_features.py \
  --input-manifest data/processed/meld/train_audio.jsonl \
  --output-manifest data/processed/meld/train.jsonl \
  --feature-dir data/features/video/meld/train \
  --num-frames 8 \
  --device cuda
```

Validation:

```bash
PYTHONPATH=src python scripts/extract_audio_features.py \
  --input-manifest data/processed/meld/val_raw.jsonl \
  --output-manifest data/processed/meld/val_audio.jsonl \
  --feature-dir data/features/audio/meld/val \
  --device cuda
```

```bash
PYTHONPATH=src python scripts/extract_video_features.py \
  --input-manifest data/processed/meld/val_audio.jsonl \
  --output-manifest data/processed/meld/val.jsonl \
  --feature-dir data/features/video/meld/val \
  --num-frames 8 \
  --device cuda
```

Test:

```bash
PYTHONPATH=src python scripts/extract_audio_features.py \
  --input-manifest data/processed/meld/test_raw.jsonl \
  --output-manifest data/processed/meld/test_audio.jsonl \
  --feature-dir data/features/audio/meld/test \
  --device cuda
```

```bash
PYTHONPATH=src python scripts/extract_video_features.py \
  --input-manifest data/processed/meld/test_audio.jsonl \
  --output-manifest data/processed/meld/test.jsonl \
  --feature-dir data/features/video/meld/test \
  --num-frames 8 \
  --device cuda
```

## Manifest Validation

```bash
PYTHONPATH=src python scripts/validate_manifest.py \
  --manifest data/processed/meld/train.jsonl \
  --labels neutral joy sadness anger fear disgust surprise
```

```bash
PYTHONPATH=src python scripts/validate_manifest.py \
  --manifest data/processed/meld/val.jsonl \
  --labels neutral joy sadness anger fear disgust surprise
```

```bash
PYTHONPATH=src python scripts/validate_manifest.py \
  --manifest data/processed/meld/test.jsonl \
  --labels neutral joy sadness anger fear disgust surprise
```

## Optional Local Training

Small smoke train:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/default.json \
  --train-manifest data/processed/meld/train_small.jsonl \
  --val-manifest data/processed/meld/val_small.jsonl \
  --output-dir artifacts/runs/meld_small_smoke
```

Full MELD baseline train:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/default.json \
  --train-manifest data/processed/meld/train.jsonl \
  --val-manifest data/processed/meld/val.jsonl \
  --output-dir artifacts/runs/meld_baseline
```

## Evaluation

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/default.json \
  --checkpoint artifacts/runs/meld_baseline/best_model.pt \
  --manifest data/processed/meld/test.jsonl \
  --output-dir artifacts/evals/meld_baseline_test
```

## Troubleshooting

- `missing_video = true`: verify the split folder and MELD clip naming. This workflow expects `dia{Dialogue_ID}_utt{Utterance_ID}.mp4`.
- Empty `audio_path`: pass `--audio-root` only if you have extracted audio files with matching stems such as `dia0_utt0.wav`.
- CUDA errors: rerun feature extraction with `--device cpu`.
- Label warnings: only `neutral`, `joy`, `sadness`, `anger`, `fear`, `disgust`, and `surprise` are kept.
- Path errors on Windows PowerShell: either use one-line commands or replace Bash line continuations (`\`) with PowerShell backticks.
- Missing script errors for later steps: confirm the feature extraction, validation, training, and evaluation scripts exist in your current branch before running those documented commands.
