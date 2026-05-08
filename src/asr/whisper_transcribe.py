from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from src.common.logging_utils import get_logger

LOGGER = get_logger(__name__)


def build_whisper_pipeline(model_name: str = "openai/whisper-large-v3"):
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, torch_dtype=dtype)
    processor = AutoProcessor.from_pretrained(model_name)
    asr_pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )
    return asr_pipe


def transcribe_manifest(manifest_path: str, output_path: str, batch_size: int = 8) -> pd.DataFrame:
    df = pd.read_parquet(manifest_path)
    asr_pipe = build_whisper_pipeline()

    transcripts = []
    conf_mean = []
    conf_min = []

    for audio_path in df["audio_path_16khz"].astype(str).tolist():
        result = asr_pipe(
            audio_path,
            batch_size=batch_size,
            generate_kwargs={"task": "transcribe", "language": "en"},
            return_timestamps=False,
        )
        text = result.get("text", "").strip()
        score = float(result.get("score", 0.5))
        transcripts.append(text)
        conf_mean.append(score)
        conf_min.append(score)

    df["transcript_text"] = transcripts
    df["transcript_source"] = "asr"
    df["asr_conf_mean"] = conf_mean
    df["asr_conf_min"] = conf_min
    df["asr_low_conf_flag"] = df["asr_conf_mean"] < 0.6
    df["whisper_language"] = "en"
    df["non_english_flag"] = False

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    LOGGER.info("Saved ASR manifest: %s", output_path)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Whisper ASR over an utterance manifest.")
    parser.add_argument("--manifest", default="data/processed/asr/utterance_manifest.parquet")
    parser.add_argument("--output", default="data/interim/transcripts_asr/utterance_asr.parquet")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transcribe_manifest(args.manifest, args.output, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
