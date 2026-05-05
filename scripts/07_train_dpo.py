#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train Qwen/Llama with LoRA DPO on RAG preference data.

Input JSONL format produced by scripts/06_build_dpo_data.py:
{
  "prompt": "...",
  "chosen": "Answer: ...\\nEvidence: ...",
  "rejected": "Answer: Insufficient evidence\\nEvidence: ..."
}

Recommended debug run:
  shuf -n 1000 data/preference/dpo_train_top10_full.jsonl \
    > data/preference/dpo_train_top10_full_debug1k.jsonl

  shuf -n 1000 data/preference/dpo_validation_top10_full.jsonl \
    > data/preference/dpo_validation_top10_full_debug1k.jsonl

  CUDA_VISIBLE_DEVICES=0 python -u scripts/07_train_dpo.py \
    --train_file data/preference/dpo_train_top10_full_debug1k.jsonl \
    --eval_file data/preference/dpo_validation_top10_full_debug1k.jsonl \
    --output_dir outputs/dpo/qwen2_5_7b_rag_dpo_debug1k \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --num_train_epochs 1 \
    --max_prompt_length 4096 \
    --max_length 4608 \
    --beta 0.1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --bf16

Recommended full run:
  CUDA_VISIBLE_DEVICES=0 python -u scripts/07_train_dpo.py \
    --train_file data/preference/dpo_train_top10_full.jsonl \
    --eval_file data/preference/dpo_val_top10_full.jsonl \
    --output_dir outputs/dpo/qwen2_5_7b_rag_dpo_top10_full \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --num_train_epochs 1 \
    --max_prompt_length 4096 \
    --max_length 4608 \
    --beta 0.1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --bf16 \
    --save_steps 500 \
    --eval_steps 500

Optional QLoRA run if memory is tight:
  CUDA_VISIBLE_DEVICES=0 python -u scripts/07_train_dpo.py \
    --train_file data/preference/dpo_train_top10_full.jsonl \
    --eval_file data/preference/dpo_val_top10_full.jsonl \
    --output_dir outputs/dpo/qwen2_5_7b_rag_dpo_top10_full_qlora \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --num_train_epochs 1 \
    --max_prompt_length 4096 \
    --max_length 4608 \
    --beta 0.1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --load_in_4bit
"""

import argparse
import inspect
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def make_json_serializable(obj):
    """Convert Path and other non-JSON objects to JSON-serializable values."""
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]

    return obj


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    obj = make_json_serializable(obj)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)



def build_dataset(
    jsonl_path: Path,
    max_examples: Optional[int],
    seed: int,
    use_chat_format: bool,
) -> Dataset:
    records = read_jsonl(jsonl_path)

    if max_examples is not None:
        random.Random(seed).shuffle(records)
        records = records[:max_examples]

    rows = []

    skipped = 0

    for record in records:
        prompt = str(record.get("prompt", "")).strip()
        chosen = str(record.get("chosen", "")).strip()
        rejected = str(record.get("rejected", "")).strip()

        if not prompt or not chosen or not rejected:
            skipped += 1
            continue

        if chosen == rejected:
            skipped += 1
            continue

        if use_chat_format:
            rows.append(
                {
                    "prompt": [{"role": "user", "content": prompt}],
                    "chosen": [{"role": "assistant", "content": chosen}],
                    "rejected": [{"role": "assistant", "content": rejected}],
                }
            )
        else:
            rows.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

    print(f"Loaded {len(rows)} usable examples from {jsonl_path}")
    print(f"Skipped examples: {skipped}")

    return Dataset.from_list(rows)


def split_train_eval(
    train_dataset: Dataset,
    eval_ratio: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    if eval_ratio <= 0:
        return train_dataset, Dataset.from_list([])

    split = train_dataset.train_test_split(
        test_size=eval_ratio,
        seed=seed,
        shuffle=True,
    )

    return split["train"], split["test"]


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DPOTrainer expects left padding for decoder-only LMs.
    tokenizer.padding_side = "left"

    return tokenizer


def load_model(args):
    model_kwargs = {
        "trust_remote_code": True,
    }

    if args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        # For single-GPU QLoRA. Use CUDA_VISIBLE_DEVICES=0 outside.
        model_kwargs["device_map"] = {"": 0}

    elif args.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model_kwargs["device_map"] = {"": 0}

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs,
    )

    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    if args.load_in_4bit or args.load_in_8bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    return model


def build_lora_config(args) -> LoraConfig:
    target_modules = [
        x.strip()
        for x in args.lora_target_modules.split(",")
        if x.strip()
    ]

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def filter_kwargs_for_dataclass(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep this script robust across TRL/Transformers versions by dropping
    unsupported DPOConfig fields.
    """
    if hasattr(cls, "__dataclass_fields__"):
        valid = set(cls.__dataclass_fields__.keys())
        return {k: v for k, v in kwargs.items() if k in valid}

    signature = inspect.signature(cls.__init__)
    valid = set(signature.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in valid}


def build_dpo_config(args) -> DPOConfig:
    config_kwargs = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": args.overwrite_output_dir,

        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,

        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,

        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,

        # Different versions may accept eval_strategy or evaluation_strategy.
        "eval_strategy": args.eval_strategy,
        "evaluation_strategy": args.eval_strategy,
        "eval_steps": args.eval_steps,

        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,

        "report_to": [] if args.report_to == "none" else [args.report_to],
        "remove_unused_columns": False,
        "seed": args.seed,

        # DPO-specific.
        "beta": args.beta,
        "loss_type": args.loss_type,
        "label_smoothing": args.label_smoothing,

        # Sequence lengths.
        "max_prompt_length": args.max_prompt_length,
        "max_length": args.max_length,

        # Optional memory/speed.
        "precompute_ref_log_probs": args.precompute_ref_log_probs,
    }

    filtered = filter_kwargs_for_dataclass(DPOConfig, config_kwargs)

    dropped = sorted(set(config_kwargs) - set(filtered))
    if dropped:
        print(f"Dropped unsupported DPOConfig kwargs for this TRL version: {dropped}")

    return DPOConfig(**filtered)


def build_dpo_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_args,
    peft_config,
):
    """
    TRL versions differ slightly:
      newer versions use processing_class=tokenizer
      older versions use tokenizer=tokenizer
    Try newer API first, then fallback.
    """
    common_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "peft_config": peft_config,
    }

    if eval_dataset is not None and len(eval_dataset) > 0:
        common_kwargs["eval_dataset"] = eval_dataset

    try:
        return DPOTrainer(
            **common_kwargs,
            processing_class=tokenizer,
        )
    except TypeError as exc:
        print(f"Falling back to tokenizer= API because processing_class failed: {exc}")
        return DPOTrainer(
            **common_kwargs,
            tokenizer=tokenizer,
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    # Data.
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--eval_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--eval_ratio", type=float, default=0.0)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_eval_examples", type=int, default=None)
    parser.add_argument(
        "--standard_format",
        action="store_true",
        help="Use plain string prompt/chosen/rejected instead of chat-format messages.",
    )

    # Model.
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help='Optional: "flash_attention_2", "sdpa", or leave unset.',
    )

    # Quantization.
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument(
        "--bnb_4bit_quant_type",
        type=str,
        default="nf4",
        choices=["nf4", "fp4"],
    )

    # LoRA.
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    # DPO / length.
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="sigmoid")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_length", type=int, default=4608)
    parser.add_argument("--precompute_ref_log_probs", action="store_true")

    # Training.
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # Logging / saving.
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_strategy", type=str, default="steps")
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--overwrite_output_dir", action="store_true")

    # Runtime.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()

    if args.bf16 and args.fp16:
        raise ValueError("Use only one of --bf16 or --fp16.")

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Use only one of --load_in_4bit or --load_in_8bit.")

    set_seed(args.seed)

    use_chat_format = not args.standard_format

    print("Building datasets...")
    train_dataset = build_dataset(
        jsonl_path=args.train_file,
        max_examples=args.max_train_examples,
        seed=args.seed,
        use_chat_format=use_chat_format,
    )

    if args.eval_file is not None:
        eval_dataset = build_dataset(
            jsonl_path=args.eval_file,
            max_examples=args.max_eval_examples,
            seed=args.seed,
            use_chat_format=use_chat_format,
        )
    elif args.eval_ratio > 0:
        train_dataset, eval_dataset = split_train_eval(
            train_dataset=train_dataset,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
        )
    else:
        eval_dataset = Dataset.from_list([])

    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")

    tokenizer = load_tokenizer(args.model_name)
    model = load_model(args)
    peft_config = build_lora_config(args)
    training_args = build_dpo_config(args)

    save_json(
        vars(args),
        Path(args.output_dir) / "train_args.json",
    )

    trainer = build_dpo_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args,
        peft_config=peft_config,
    )

    print("Starting DPO training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("Saving final model / adapter...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Saved DPO output to: {args.output_dir}")


if __name__ == "__main__":
    main()
