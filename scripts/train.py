"""Обучение LoRA r=16 в двух режимах: bf16-база и nf4 (QLoRA).

Примеры:
    python scripts/train.py --mode bf16 --out runs/bf16
    python scripts/train.py --mode nf4  --out runs/nf4
    python scripts/train.py --mode bf16 --out runs/bf16 --model D:/models/Qwen2.5-1.5B-Instruct

Время обучения и пиковая память GPU пишутся в <out>/train_meta.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import MAX_SEQ_LENGTH, SEED, MODEL_NAME, RUNS_DIR, DATA_DIR, ensure_dirs  # noqa: E402
from data import load_rows_from_jsonl, to_sft_dataset  # noqa: E402

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _build_model(model_name: str, load_in_4bit: bool):
    """Базовая модель (bf16 или NF4) + LoRA r=16."""
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        dtype=torch.bfloat16 if not load_in_4bit else None,
        device_map={"": 0},
    )

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bf16", "nf4"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--model", default=MODEL_NAME, help="Путь к локальной папке с весами или HF-имя")
    args = ap.parse_args()

    ensure_dirs()
    out = Path(args.out or (RUNS_DIR / args.mode))
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _build_model(args.model, args.mode == "nf4")

    train_rows = load_rows_from_jsonl(DATA_DIR / "train.jsonl")
    dataset = to_sft_dataset(train_rows, tokenizer)
    tokenizer.padding_side = "right"

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    targs = TrainingArguments(
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_steps=5,
        learning_rate=args.lr,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="no",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=SEED,
        output_dir=str(out),
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        data_collator=data_collator,
        args=targs,
    )
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    stats = trainer.train()
    train_time = time.perf_counter() - t0

    model.save_pretrained(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))

    meta = {
        "mode": args.mode,
        "model": args.model,
        "r": 16,
        "alpha": 16,
        "epochs": args.epochs,
        "batch": args.batch,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "seed": SEED,
        "train_time_s": round(train_time, 3),
        "train_steps": int(stats.global_step),
        "peak_mem_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
