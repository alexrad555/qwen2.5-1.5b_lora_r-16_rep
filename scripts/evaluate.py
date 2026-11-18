"""Качество по сырым ответам: accuracy, macro-F1, доля разобранных JSON.

Неразобранные ответы не выбрасываются - считаются ошибкой,
иначе метрики завышаются (см. замечание руководителя про обрезку).

Запуск:
    python scripts/evaluate.py                    # все прогоны в runs/
    python scripts/evaluate.py --run runs/inf_bf16_base
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import LABELS, RESULTS_DIR  # noqa: E402

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_answer(raw: str) -> dict | None:
    """Достаёт JSON из ответа: как есть -> без ```-заборов -> первое {...}."""
    text = raw.strip()
    candidates = [text, text.strip("`").strip()]
    m = JSON_RE.search(text)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and "label" in obj:
            return obj
    return None


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    parsed = 0
    correct = 0
    f1s = []
    for lab in LABELS:
        tp = fp = fn = 0
        for r in rows:
            obj = parse_answer(r["raw"])
            pred = str(obj["label"]).strip().lower() if obj else None
            if obj:
                parsed += 1
            if pred == lab and r["label"] == lab:
                tp += 1
            elif pred == lab:
                fp += 1
            elif r["label"] == lab:
                fn += 1
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    correct = sum(1 for r in rows
                  if (o := parse_answer(r["raw"])) and
                  str(o["label"]).strip().lower() == r["label"])
    return {
        "n": n,
        "accuracy": round(correct / n, 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "json_ok": round(parsed / n, 4) if n else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="Конкретная папка прогона (по умолчанию все в runs/)")
    args = ap.parse_args()

    run_dirs = [Path(args.run)] if args.run else sorted(
        d for d in (ROOT / "runs").iterdir() if d.is_dir() and d.name.startswith("inf_"))

    table = []
    for run_dir in run_dirs:
        for raw_path in sorted(run_dir.glob("raw_*_all.jsonl")):
            rows = [json.loads(line) for line in open(raw_path, encoding="utf-8") if line.strip()]
            m = metrics(rows)
            m["run"] = f"{run_dir.name}/{raw_path.stem}"
            table.append(m)

    if not table:
        print("Нет сырых ответов: сначала запустите scripts/inference.py")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    out_csv = RESULTS_DIR / "metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "n", "accuracy", "macro_f1", "json_ok"])
        w.writeheader()
        w.writerows(table)

    print(f"{'прогон':45} {'n':>4} {'acc':>7} {'f1':>7} {'json':>7}")
    for m in table:
        print(f"{m['run']:45} {m['n']:>4} {m['accuracy']:>7.3f} {m['macro_f1']:>7.3f} {m['json_ok']:>7.3f}")
    print(f"\nCSV: {out_csv}")


if __name__ == "__main__":
    main()
