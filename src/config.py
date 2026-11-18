"""Сиды, пути и основные константы."""
from __future__ import annotations

import os
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs"

MODEL_NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

LABELS = ["negative", "neutral", "positive"]

# Ответ модели - короткий JSON, но лимит берём с запасом, чтобы ничего не обрезалось.
MAX_NEW_TOKENS = 512
MAX_SEQ_LENGTH = 2048


def ensure_dirs() -> None:
    for d in (DATA_DIR, RESULTS_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
