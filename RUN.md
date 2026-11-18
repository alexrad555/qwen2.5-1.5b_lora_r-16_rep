# Как запустить

Окружение (один раз):

```
py -3.12 -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\pip install -r requirements.txt
```
Данные (сплит seed=42 уже лежит в data/, пересоздавать не нужно).

Обучение двух адаптеров (по ~30-60 мин на RTX 4090):


```
.venv\Scripts\python scripts\train.py --mode bf16 --out runs\bf16
.venv\Scripts\python scripts\train.py --mode nf4  --out runs\nf4
```

Инференс: каждый адаптер на трёх квантизациях базы (6 прогонов, ~2-3 ч):

```
.venv\Scripts\python scripts\inference.py --adapter runs\bf16\adapter --quant bf16 --out runs\inf_bf16_base
.venv\Scripts\python scripts\inference.py --adapter runs\bf16\adapter --quant 8bit --out runs\inf_bf16_8bit
.venv\Scripts\python scripts\inference.py --adapter runs\bf16\adapter --quant nf4  --out runs\inf_bf16_nf4
.venv\Scripts\python scripts\inference.py --adapter runs\nf4\adapter --quant bf16 --out runs\inf_nf4_base
.venv\Scripts\python scripts\inference.py --adapter runs\nf4\adapter --quant 8bit --out runs\inf_nf4_8bit
.venv\Scripts\python scripts\inference.py --adapter runs\nf4\adapter --quant nf4  --out runs\inf_nf4_nf4
```

Метрики качества (accuracy, macro-F1, доля разобранных JSON):

```
.venv\Scripts\python scripts\evaluate.py
```

Результаты: results\metrics.csv
сырые ответы — runs\inf_*\raw_*_all.jsonl
скорость/память — runs\inf_*\bench_*.json,
обучение — runs\*\train_meta.json

Первый запуск скачает модель Qwen2.5-1.5B-Instruct (~3 ГБ) с huggingface.co.
Если сеть до HuggingFace нестабильна, можно скачать модель один раз вручную
и передавать путь: --model D:\models\Qwen2.5-1.5B-Instruct

выводы написаны в report.docx
