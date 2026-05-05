#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate RAG answers with a DPO LoRA adapter.

Example:
  CUDA_VISIBLE_DEVICES=0 python -u scripts/08_generate_dpo.py \
    --retrieval_file data/retrieval/nq_validation_bm25_top10_full.jsonl \
    --output data/generation/dpo_val_outputs_top10_full.jsonl \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --adapter_dir outputs/dpo/qwen2_5_7b_rag_dpo_top10_full_1epoch_len3328_lr1e6 \
    --top_k 10 \
    --max_new_tokens 64 \
    --max_input_length 4096 \
    --batch_size 4
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_existing_ids(path: Path) -> Set[str]:
    existing_ids = set()

    if not path.exists():
        return existing_ids

    for record in read_jsonl(path):
        if "id" in record:
            existing_ids.add(str(record["id"]))

    return existing_ids


def batch_iter(records: List[Dict[str, Any]], batch_size: int):
    for i in range(0, len(records), batch_size):
        yield records[i : i + batch_size]


def format_passages(passages: List[Dict[str, Any]], top_k: int) -> str:
    lines = []

    for i, passage in enumerate(passages[:top_k], start=1):
        pid = passage.get("rank", i)
        title = str(passage.get("title", "")).strip()
        text = str(passage.get("text", "")).strip()

        if title:
            lines.append(f"[{pid}] {title}\n{text}")
        else:
            lines.append(f"[{pid}] {text}")

    return "\n\n".join(lines)


def build_prompt(question: str, passages: List[Dict[str, Any]], top_k: int) -> str:
    context = format_passages(passages, top_k=top_k)

    return f"""You are a factual retrieval-augmented question answering assistant.

Answer the question using only the provided passages. Do not use prior knowledge.

When the answer is explicitly stated or can be directly inferred from the passages, give the shortest correct answer possible.
Then give one brief evidence sentence with citations.
Only cite passages that support the answer.

If the passages do not contain enough information to infer the answer, write "Insufficient evidence" as the short answer.

Output format:
Answer: <short answer>
Evidence: <one brief sentence with citations>

Question:
{question}

Passages:
{context}

Now produce the answer in the required format."""


def load_model_and_tokenizer(model_name: str, adapter_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
    )

    model.eval()
    return model, tokenizer


def build_chat_inputs(tokenizer, prompts: List[str]) -> List[str]:
    model_inputs = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt

        model_inputs.append(text)

    return model_inputs


@torch.no_grad()
def generate_answers_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    max_input_length: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    model_inputs = build_chat_inputs(tokenizer, prompts)

    inputs = tokenizer(
        model_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    outputs = model.generate(
        **inputs,
        **generation_kwargs,
    )

    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_length:]

    answers = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return [answer.strip() for answer in answers]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_dir", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_input_length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    records = list(read_jsonl(args.retrieval_file))

    if args.max_examples is not None:
        records = records[: args.max_examples]

    print(f"Loaded {len(records)} retrieval examples")
    print(f"Base model: {args.model_name}")
    print(f"Adapter: {args.adapter_dir}")
    print(
        f"top_k={args.top_k}, "
        f"max_new_tokens={args.max_new_tokens}, "
        f"max_input_length={args.max_input_length}, "
        f"batch_size={args.batch_size}"
    )

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        adapter_dir=args.adapter_dir,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = set()
    write_mode = "w"

    if args.resume and args.output.exists():
        existing_ids = load_existing_ids(args.output)
        write_mode = "a"
        print(f"Resume mode: found {len(existing_ids)} existing examples in {args.output}")

    filtered_records = []
    for record in records:
        record_id = str(record.get("id"))
        if args.resume and record_id in existing_ids:
            continue
        filtered_records.append(record)

    print(f"Examples to generate: {len(filtered_records)}")

    num_written = 0
    start_time = time.time()

    with args.output.open(write_mode, encoding="utf-8") as fout:
        for batch in tqdm(
            batch_iter(filtered_records, args.batch_size),
            total=(len(filtered_records) + args.batch_size - 1) // args.batch_size,
            desc="Generating DPO answers",
        ):
            prompts = []

            for record in batch:
                question = str(record["question"])
                retrieved_passages = record.get("retrieved_passages", [])

                prompt = build_prompt(
                    question=question,
                    passages=retrieved_passages,
                    top_k=args.top_k,
                )

                prompts.append(prompt)

            generated_answers = generate_answers_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                max_input_length=args.max_input_length,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            for record, prompt, generated_answer in zip(batch, prompts, generated_answers):
                output_record = {
                    "id": record.get("id"),
                    "question": str(record["question"]),
                    "answers": record.get("answers", []),
                    "retrieved_passages": record.get("retrieved_passages", [])[: args.top_k],
                    "prompt": prompt,
                    "generated_answer": generated_answer,
                    "model_name": args.model_name,
                    "adapter_dir": str(args.adapter_dir),
                    "top_k": args.top_k,
                    "max_new_tokens": args.max_new_tokens,
                    "max_input_length": args.max_input_length,
                }

                fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                num_written += 1

            fout.flush()

    elapsed = time.time() - start_time
    speed = num_written / elapsed if elapsed > 0 else 0.0

    print(f"Saved DPO generations to {args.output}")
    print(f"Written examples: {num_written}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Speed: {speed:.2f} examples/sec")


if __name__ == "__main__":
    main()
