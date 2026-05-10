# Multimodal Emotion Recognition

Single-GPU, offline/batch multimodal emotion recognition prototype.

## Scope
- Input: video file
- Modalities: audio, video frames, text (ASR or oracle transcript)
- Outputs:
  - Temporal timeline with per-window emotion predictions
  - Duration-weighted emotion distribution across full video

## Label space
The project uses the MELD 7-class taxonomy in this fixed order:
- neutral
- surprise
- fear
- sadness
- joy
- disgust
- anger

## Pipeline overview
1. Prepare manifests and labels
2. Build leakage-safe CV splits (grouped)
3. Train unimodal branches (audio, video, text)
4. Train late-fusion model
5. Calibrate with temperature scaling
6. Evaluate on ablations and CMU-MOSEI generalization
7. Export ONNX models for batch serving

## Quick start
### Local
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.cli.prepare_data --regime oracle
python -m src.cli.prepare_data --regime asr
python -m src.cli.train_unimodal --branch audio --regime asr --fold 0
python -m src.cli.train_unimodal --branch video --regime asr --fold 0
python -m src.cli.train_unimodal --branch text --regime asr --fold 0
python -m src.cli.train_fusion --mode late --regime asr --fold 0
python -m src.cli.evaluate --suite full --regime asr
```

### Colab workflow
1. Upload or clone this repository in Colab.
2. Install dependencies from `requirements-colab.txt`.
3. Mount Drive and set `DATA_ROOT` to your dataset location.
4. Use `scripts/colab_train.sh` to run fold training.

## Data split policy
- MELD grouped by dialogue_id
- IEMOCAP grouped by session_id
- StratifiedGroupKFold with 5 folds, seed 42
- CMU-MOSEI is test-only and never used for train/val

## Transcript regimes
- oracle: dataset annotations
- asr: Whisper large-v3 outputs

These regimes are always stored, trained, and evaluated separately.
