from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(obj, handle, sort_keys=False)


def write_json(obj: Dict[str, Any], path: str | Path) -> None:
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=True, indent=2)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    ensure_parent(path)
    df.to_parquet(path, index=False)
