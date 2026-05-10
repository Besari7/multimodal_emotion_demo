from jsonschema import validate

from src.serving.schema_validate import DISTRIBUTION_SCHEMA, TIMELINE_SCHEMA


def test_timeline_schema_sample() -> None:
    sample = {
        "video_id": "vid1",
        "model_version": "0.1.0",
        "transcript_regime": "asr",
        "window_size_sec": 5.0,
        "window_stride_sec": 2.5,
        "class_order": ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"],
        "timeline": [
            {
                "window_index": 0,
                "start_time_sec": 0.0,
                "end_time_sec": 5.0,
                "effective_duration_sec": 5.0,
                "is_padded": False,
                "pad_right_sec": 0.0,
                "predicted_label": "neutral",
                "probability_vector": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "confidence": 1.0,
                "missing_modality_flags": {"audio": False, "video": False, "text": False},
            }
        ],
    }
    validate(instance=sample, schema=TIMELINE_SCHEMA)


def test_distribution_schema_sample() -> None:
    sample = {
        "video_id": "vid1",
        "aggregation": "duration_weighted",
        "total_effective_duration_sec": 5.0,
        "distribution": [
            {"label": "neutral", "duration_sec": 5.0, "percentage": 100.0, "soft_percentage": 100.0}
        ],
        "dominant_label": "neutral",
    }
    validate(instance=sample, schema=DISTRIBUTION_SCHEMA)
