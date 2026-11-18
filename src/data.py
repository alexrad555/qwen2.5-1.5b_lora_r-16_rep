"""Промпт, формат ответа и подготовка данных для обучения/инференса."""
from __future__ import annotations

import json
from pathlib import Path

from config import LABELS, MAX_SEQ_LENGTH

SYSTEM_PROMPT = (
    "Ты — анализатор тональности русскоязычных отзывов о товарах. "
    "Прочитай отзыв покупателя и оцени его тональность. "
    "Ответь строго одним JSON-объектом без markdown-разметки и без лишнего текста в виде:\n"
    '{"label": "<negative|neutral|positive>", "explanation": "<короткое пояснение на русском, 1 предложение>"}'
)


def to_json(label: str, explanation: str) -> str:
    return json.dumps({"label": label, "explanation": explanation}, ensure_ascii=False)


def build_example(text: str, label: str, explanation: str) -> dict:
    return {"text": text.strip(), "label": label, "explanation": explanation.strip()}


def format_user(text: str) -> str:
    return f"Обращение: {text.strip()}"


def format_assistant(label: str, explanation: str) -> str:
    return to_json(label, explanation)


def load_rows_from_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_sft_dataset(rows: list[dict], tokenizer):
    """Токенизированный датасет для SFT: лосс только на ответе ассистента.

    Промпт (system + user) маскируется меткой -100, иначе модель учится
    генерировать промпты, а не ответы.
    """
    from datasets import Dataset

    def _fmt(r: dict) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user(r["text"])},
            {"role": "assistant", "content": format_assistant(r["label"], r["explanation"])},
        ]
        full = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
                                               add_generation_prompt=True)
        full_ids = tokenizer(full, truncation=True, max_length=MAX_SEQ_LENGTH)["input_ids"]
        prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_SEQ_LENGTH)["input_ids"]

        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels[:prompt_len] = [-100] * prompt_len
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    return Dataset.from_list([_fmt(r) for r in rows])
