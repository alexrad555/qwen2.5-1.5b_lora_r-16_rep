"""Инференс одного LoRA-адаптера на квантованной базовой модели.

Режимы квантизации базы: bf16 / 8bit / nf4. Для каждого размера батча (1/8/32)
считаются: время прогона, tok/s, пиковая память GPU и задержка на batch=1.

Запуск:
    python scripts/inference.py --adapter runs/bf16/adapter --quant bf16 --out runs/inf_bf16_base
    python scripts/inference.py --adapter runs/bf16/adapter --quant 8bit --out runs/inf_bf16_8bit
    python scripts/inference.py --adapter runs/bf16/adapter --quant nf4  --out runs/inf_bf16_nf4
    python scripts/inference.py --adapter runs/nf4/adapter  --quant bf16 --out runs/inf_nf4_base
    ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import (  # noqa: E402
    MAX_NEW_TOKENS, MAX_SEQ_LENGTH, MODEL_NAME, DATA_DIR, ensure_dirs,
)
from data import SYSTEM_PROMPT, format_user, load_rows_from_jsonl  # noqa: E402

QUANTS = ["bf16", "8bit", "nf4"]
BATCHES = [1, 8, 32]


def build_model(model_name: str, quant: str):
    """Базовая модель в нужном режиме квантизации."""
    quantization_config = None
    if quant == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "nf4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        dtype=torch.bfloat16 if quant == "bf16" else None,
        device_map={"": 0},
    )
    return model


def build_prompts(rows: list[dict], tokenizer) -> list[str]:
    prompts = []
    for r in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user(r["text"])},
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True))
    return prompts


@torch.no_grad()
def generate(model, tokenizer, prompts: list[str]) -> list[str]:
    """Генерация ответов для батча промптов (только новые токены, без промпта)."""
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    gen = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    n_prompt = enc["input_ids"].shape[1]
    return tokenizer.batch_decode(gen[:, n_prompt:], skip_special_tokens=True)


def run_config(model, tokenizer, rows, quant: str, out_dir: Path) -> dict:
    prompts_all = build_prompts(rows, tokenizer)
    result = {"quant": quant, "batches": {}}

    for bs in BATCHES:
        batches = [prompts_all[i:i + bs] for i in range(0, len(prompts_all), bs)]
        torch.cuda.reset_peak_memory_stats()
        all_out = []
        t0 = time.perf_counter()
        for chunk in tqdm(batches, desc=f"{quant} | batch={bs}", unit="батч"):
            all_out.extend(generate(model, tokenizer, chunk))
        dt = time.perf_counter() - t0
        total_tokens = sum(len(tokenizer(o, add_special_tokens=False)["input_ids"])
                           for o in all_out)
        result["batches"][str(bs)] = {
            "n": len(all_out),
            "time_s": round(dt, 3),
            "throughput_tok_s": round(total_tokens / dt, 2) if dt > 0 else 0.0,
            "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        }
    
        if bs == BATCHES[-1]:
            with open(out_dir / f"raw_{quant}_all.jsonl", "w", encoding="utf-8") as f:
                for r, o in zip(rows, all_out):
                    f.write(json.dumps({"text": r["text"], "label": r["label"],
                                        "explanation": r["explanation"], "raw": o},
                                       ensure_ascii=False) + "\n")

    b1 = result["batches"]["1"]
    result["latency_ms"] = round(b1["time_s"] / b1["n"] * 1000, 2)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Папка с адаптером (runs/.../adapter)")
    ap.add_argument("--quant", choices=QUANTS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Ограничить число примеров")
    ap.add_argument("--model", default=MODEL_NAME, help="Путь к базовой модели или HF-имя")
    args = ap.parse_args()

    ensure_dirs()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 

    base = build_model(args.model, args.quant)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    rows = load_rows_from_jsonl(DATA_DIR / "test.jsonl")
    if args.limit:
        rows = rows[: args.limit]

    result = run_config(model, tokenizer, rows, args.quant, out_dir)
    result["adapter"] = args.adapter
    result["n_test"] = len(rows)
    (out_dir / f"bench_{args.quant}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
