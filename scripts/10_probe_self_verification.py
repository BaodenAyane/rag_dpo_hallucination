#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Probe whether the model can recognize unsupported citations under an explicit
verification prompt.

This script selects examples where a target system, usually dpo_v1, was judged
to have unfaithful citations by the external LLM verifier. It then asks base,
DPO-v1, and DPO-v2 models to verify whether the cited passages support the
candidate answer.

Example:
  python scripts/10_probe_self_verification.py \
    --audit_file results/llm_verifier_audit_val_all_ecnu_v2.jsonl \
    --output_file results/self_verify_dpo_v1_unfaithful_sample200.jsonl \
    --target_system dpo_v1 \
    --sample_size 200 \
    --base_url http://localhost:8000/v1 \
    --api_key local-vllm \
    --models Qwen/Qwen2.5-7B-Instruct dpo_v1 dpo_v2
"""

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


SYSTEM_PROMPT = """You are a strict verifier for retrieval-augmented question answering.

You will be given:
- a question,
- a candidate answer,
- cited passages.

Your task is to decide whether the cited passages support the candidate answer.

Rules:
1. Use only the cited passages.
2. Do not use outside knowledge.
3. If the candidate answer is "Insufficient evidence", then cited passages should not be treated as supporting citations.
4. Return valid JSON only.
"""


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_citations(text: Any) -> List[int]:
    return sorted(set(int(x) for x in re.findall(r"\[(\d+)\]", str(text or ""))))


def normalize_text(x: Any) -> str:
    return str(x or "").strip()


def get_cited_passages(row: Dict[str, Any], generated_answer: str) -> List[Dict[str, Any]]:
    citation_ids = parse_citations(generated_answer)
    passages = row.get("retrieved_passages", [])

    by_rank = {}
    for idx, p in enumerate(passages, start=1):
        try:
            rank = int(p.get("rank", idx))
        except Exception:
            rank = idx
        by_rank[rank] = p

    cited = []
    for cid in citation_ids:
        p = by_rank.get(cid)
        if p is None:
            continue
        cited.append(
            {
                "rank": cid,
                "title": normalize_text(p.get("title")),
                "text": normalize_text(p.get("text")),
            }
        )

    return cited


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {
            "supported": "parse_error",
            "reason": text[:500],
        }

    try:
        return json.loads(m.group(0))
    except Exception:
        return {
            "supported": "parse_error",
            "reason": text[:500],
        }


def build_prompt(question: str, candidate_answer: str, cited_passages: List[Dict[str, Any]]) -> str:
    payload = {
        "question": question,
        "candidate_answer": candidate_answer,
        "cited_passages": cited_passages,
        "task": (
            "Do the cited passages support the candidate answer? "
            "Return yes, partial, or no."
        ),
        "output_format": {
            "supported": "yes | partial | no",
            "reason": "brief explanation",
            "unsupported_citation_ids": ["list of citation ranks that do not support the answer"],
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    max_retries: int = 3,
    sleep_seconds: float = 1.0,
) -> Dict[str, Any]:
    last_error = None

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=256,
            )
            content = resp.choices[0].message.content or ""
            out = extract_json(content)

            supported = str(out.get("supported", "")).strip().lower()
            if supported not in {"yes", "partial", "no", "parse_error"}:
                out["supported_original"] = supported
                out["supported"] = "parse_error"

            return out

        except Exception as e:
            last_error = e
            time.sleep(sleep_seconds * (2 ** attempt))

    return {
        "supported": "api_error",
        "reason": str(last_error),
    }


def select_examples(
    rows: List[Dict[str, Any]],
    target_system: str,
    sample_size: int,
    seed: int,
    mode: str,
) -> List[Dict[str, Any]]:
    selected = []

    for r in rows:
        judge = r.get("judge", {}).get(target_system, {})
        systems = r.get("systems", {}).get(target_system, {})
        gen = systems.get("generated_answer", "")

        if judge.get("citation_faithfulness") != "unfaithful":
            continue

        if not parse_citations(gen):
            continue

        if mode == "answer_unfaithful":
            if judge.get("answer_type") != "answer":
                continue
        elif mode == "abstention_with_citation":
            if judge.get("answer_type") != "abstention":
                continue
        elif mode == "all_unfaithful":
            pass
        else:
            raise ValueError("--mode must be answer_unfaithful, abstention_with_citation, or all_unfaithful")

        selected.append(r)

    random.seed(seed)
    random.shuffle(selected)

    if sample_size > 0:
        selected = selected[:sample_size]

    return selected


def summarize(rows: List[Dict[str, Any]], models: List[str]) -> Dict[str, Any]:
    summary = {
        "num_examples": len(rows),
        "models": {},
    }

    for model in models:
        counts = {}
        for r in rows:
            label = r.get("self_verification", {}).get(model, {}).get("supported", "missing")
            counts[label] = counts.get(label, 0) + 1

        total = len(rows)
        summary["models"][model] = {
            k: {
                "count": v,
                "rate": v / total if total else 0.0,
            }
            for k, v in sorted(counts.items())
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--summary_file", type=Path, default=None)

    parser.add_argument("--target_system", type=str, default="dpo_v1")
    parser.add_argument(
        "--mode",
        type=str,
        default="answer_unfaithful",
        choices=["answer_unfaithful", "abstention_with_citation", "all_unfaithful"],
    )

    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default="local-vllm")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen/Qwen2.5-7B-Instruct", "dpo_v1", "dpo_v2"],
    )

    args = parser.parse_args()

    rows = read_jsonl(args.audit_file)
    selected = select_examples(
        rows=rows,
        target_system=args.target_system,
        sample_size=args.sample_size,
        seed=args.seed,
        mode=args.mode,
    )

    print(f"Selected {len(selected)} examples")

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    output_rows = []

    with args.output_file.open("w", encoding="utf-8") as f:
        for idx, r in enumerate(selected, start=1):
            target_output = r["systems"][args.target_system]
            generated_answer = target_output.get("generated_answer", "")
            cited_passages = get_cited_passages(r, generated_answer)

            prompt = build_prompt(
                question=r.get("question", ""),
                candidate_answer=generated_answer,
                cited_passages=cited_passages,
            )

            self_verification = {}

            for model in args.models:
                result = call_model(
                    client=client,
                    model=model,
                    prompt=prompt,
                )
                self_verification[model] = result

            out = {
                "id": r.get("id"),
                "question": r.get("question"),
                "gold_answers": r.get("gold_answers"),
                "target_system": args.target_system,
                "target_generated_answer": generated_answer,
                "external_judge": r.get("judge", {}).get(args.target_system),
                "cited_passages": cited_passages,
                "self_verification": self_verification,
            }

            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            output_rows.append(out)

            print(f"[{idx}/{len(selected)}] id={r.get('id')}")

    summary = summarize(output_rows, args.models)

    if args.summary_file is None:
        args.summary_file = args.output_file.with_suffix(".summary.json")

    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
