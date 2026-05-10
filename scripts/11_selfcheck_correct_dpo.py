#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference-time self-check correction for DPO-v1 RAG outputs.

This script verifies whether the cited passages support the generated answer.
If the self-check model says the citations do not support the answer, the
output is conservatively rewritten as an insufficient-evidence response with
no citation.

This does NOT train a new model.

Example:

  python scripts/11_selfcheck_correct_dpo.py \
    --input_file outputs/dpo/dpo_val_eval_top10_full.jsonl \
    --output_file data/generation/dpo_v1_selfcheck_val_outputs_top10_full.jsonl \
    --summary_file results/dpo_v1_selfcheck_val_summary.json \
    --model dpo_v1 \
    --base_url http://localhost:8000/v1 \
    --api_key local-vllm \
    --correction_policy relaxed \
    --resume

Then evaluate:

  python scripts/05_eval_generation.py \
    --generation_file data/generation/dpo_v1_selfcheck_val_outputs_top10_full.jsonl \
    --output_file outputs/dpo/dpo_v1_selfcheck_val_metrics_top10_full.json \
    --per_example_output outputs/dpo/dpo_v1_selfcheck_val_eval_top10_full.jsonl
"""

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

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
4. A passage is supportive only if it directly supports the answer to the question.
5. If the cited passages are only topically related but do not support the answer, return "no".
6. Return valid JSON only.
"""


NO_EVIDENCE_RESPONSE = (
    "Answer: Insufficient evidence\n"
    "Evidence: The provided passages do not contain enough information to answer. "
    "No passage is cited because none supports the answer."
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_text(x: Any) -> str:
    return str(x or "").strip()


def parse_citations(text: Any) -> List[int]:
    return sorted(set(int(x) for x in re.findall(r"\[(\d+)\]", str(text or ""))))


def has_citations(text: Any) -> bool:
    return len(parse_citations(text)) > 0


def normalize_for_abstention(text: Any) -> str:
    x = str(text or "").lower()
    x = re.sub(r"[^a-z0-9 ]+", " ", x)
    x = " ".join(x.split())
    return x


def is_abstention(text: Any) -> bool:
    x = normalize_for_abstention(text)
    patterns = [
        "insufficient evidence",
        "not enough evidence",
        "cannot answer",
        "cant answer",
        "i dont know",
        "provided passages do not contain",
        "provided evidence is insufficient",
        "no sufficient evidence",
    ]
    return any(p in x for p in patterns)


def get_cited_passages(
    retrieved_passages: List[Dict[str, Any]],
    generated_answer: str,
    max_chars_per_passage: int,
) -> List[Dict[str, Any]]:
    citation_ids = parse_citations(generated_answer)

    by_rank: Dict[int, Dict[str, Any]] = {}
    for idx, passage in enumerate(retrieved_passages, start=1):
        try:
            rank = int(passage.get("rank", idx))
        except Exception:
            rank = idx
        by_rank[rank] = passage

    cited = []
    for cid in citation_ids:
        passage = by_rank.get(cid)
        if passage is None:
            continue

        text = normalize_text(
            passage.get("text")
            or passage.get("contents")
            or passage.get("passage")
        )

        if len(text) > max_chars_per_passage:
            text = text[:max_chars_per_passage] + " ..."

        cited.append(
            {
                "rank": cid,
                "title": normalize_text(passage.get("title")),
                "text": text,
            }
        )

    return cited


def extract_json(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {
        "supported": "parse_error",
        "reason": text[:500],
    }


def build_verification_prompt(
    question: str,
    candidate_answer: str,
    cited_passages: List[Dict[str, Any]],
) -> str:
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
            "unsupported_citation_ids": [
                "list of citation ranks that do not support the answer"
            ],
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def normalize_supported_label(value: Any) -> str:
    x = str(value or "").strip().lower()

    aliases = {
        "supported": "yes",
        "fully_supported": "yes",
        "fully supported": "yes",
        "true": "yes",
        "partially_supported": "partial",
        "partially supported": "partial",
        "not fully supported": "partial",
        "unsupported": "no",
        "not_supported": "no",
        "not supported": "no",
        "false": "no",
    }

    x = aliases.get(x, x)

    if x not in {"yes", "partial", "no", "parse_error", "api_error"}:
        return "parse_error"

    return x


def call_selfcheck(
    client: OpenAI,
    model: str,
    prompt: str,
    max_retries: int,
    sleep_seconds: float,
    max_tokens: int,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content or ""
            parsed = extract_json(content)
            parsed["raw_response"] = content
            parsed["supported"] = normalize_supported_label(parsed.get("supported"))
            return parsed

        except Exception as e:
            last_error = e
            wait = sleep_seconds * (2 ** attempt)
            print(
                f"[WARN] self-check failed "
                f"attempt={attempt + 1}/{max_retries}: {e}"
            )
            print(f"[WARN] sleeping {wait:.1f}s")
            time.sleep(wait)

    return {
        "supported": "api_error",
        "reason": str(last_error),
    }


def should_rewrite(
    supported_label: str,
    correction_policy: str,
) -> bool:
    """
    relaxed:
      rewrite only when supported == no

    strict:
      rewrite when supported == no or partial

    none:
      never rewrite, only attach self-check metadata
    """
    if correction_policy == "none":
        return False

    if correction_policy == "relaxed":
        return supported_label == "no"

    if correction_policy == "strict":
        return supported_label in {"no", "partial"}

    raise ValueError(f"Unknown correction_policy: {correction_policy}")


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    counters = Counter()
    support_counter = Counter()

    for row in rows:
        meta = row.get("selfcheck_metadata", {})

        if meta.get("had_citation"):
            counters["had_citation"] += 1
        else:
            counters["no_citation"] += 1

        if meta.get("checked"):
            counters["checked"] += 1

        if meta.get("correction_applied"):
            counters["correction_applied"] += 1

        if meta.get("original_was_abstention"):
            counters["original_was_abstention"] += 1

        if meta.get("original_was_abstention") and meta.get("had_citation"):
            counters["abstention_with_citation"] += 1

        if meta.get("original_was_abstention") and meta.get("correction_applied"):
            counters["corrected_abstention_with_citation"] += 1

        label = meta.get("selfcheck_supported")
        if label:
            support_counter[label] += 1

    def rate(x: int, denom: int = n) -> float:
        return x / denom if denom else 0.0

    checked = counters["checked"]

    return {
        "num_examples": n,
        "counts": dict(counters),
        "rates": {
            "had_citation_rate": rate(counters["had_citation"]),
            "checked_rate": rate(counters["checked"]),
            "correction_applied_rate": rate(counters["correction_applied"]),
            "abstention_with_citation_rate": rate(
                counters["abstention_with_citation"]
            ),
            "corrected_abstention_with_citation_rate": rate(
                counters["corrected_abstention_with_citation"]
            ),
        },
        "selfcheck_supported_distribution": {
            label: {
                "count": count,
                "rate_among_checked": count / checked if checked else 0.0,
                "rate_among_all": count / n if n else 0.0,
            }
            for label, count in sorted(support_counter.items())
        },
    }


def load_done_ids(output_file: Path) -> set:
    done = set()

    if not output_file.exists():
        return done

    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                done.add(str(row.get("id")))

    return done


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--summary_file", type=Path, required=True)

    parser.add_argument("--model", type=str, default="dpo_v1")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default="local-vllm")

    parser.add_argument(
        "--correction_policy",
        type=str,
        default="relaxed",
        choices=["relaxed", "strict", "none"],
        help=(
            "relaxed: rewrite only supported=no; "
            "strict: rewrite supported=no or partial; "
            "none: only attach self-check metadata"
        ),
    )

    parser.add_argument("--max_chars_per_passage", type=int, default=900)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--sleep_seconds", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    rows = read_jsonl(args.input_file)

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    done_ids = set()
    output_rows: List[Dict[str, Any]] = []

    if args.resume and args.output_file.exists():
        done_ids = load_done_ids(args.output_file)
        output_rows = read_jsonl(args.output_file)
        print(f"[resume] loaded {len(done_ids)} completed examples")
    else:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text("", encoding="utf-8")

    for idx, row in enumerate(rows, start=1):
        row_id = str(row.get("id"))

        if args.resume and row_id in done_ids:
            continue

        original_answer = row.get("generated_answer", "")
        question = row.get("question", "")
        retrieved_passages = row.get("retrieved_passages", [])

        citation_ids = parse_citations(original_answer)
        had_citation = len(citation_ids) > 0
        original_was_abstention = is_abstention(original_answer)

        corrected = dict(row)
        corrected["original_generated_answer"] = original_answer

        metadata = {
            "checked": False,
            "had_citation": had_citation,
            "citation_ids": citation_ids,
            "original_was_abstention": original_was_abstention,
            "correction_policy": args.correction_policy,
            "correction_applied": False,
        }

        if not had_citation:
            corrected["selfcheck_metadata"] = metadata
            append_jsonl(args.output_file, corrected)
            output_rows.append(corrected)
            print(f"[{idx}/{len(rows)}] id={row_id} no citation, kept")
            continue

        cited_passages = get_cited_passages(
            retrieved_passages=retrieved_passages,
            generated_answer=original_answer,
            max_chars_per_passage=args.max_chars_per_passage,
        )

        prompt = build_verification_prompt(
            question=question,
            candidate_answer=original_answer,
            cited_passages=cited_passages,
        )

        result = call_selfcheck(
            client=client,
            model=args.model,
            prompt=prompt,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
            max_tokens=args.max_tokens,
        )

        supported_label = result.get("supported", "parse_error")

        metadata.update(
            {
                "checked": True,
                "selfcheck_supported": supported_label,
                "selfcheck_result": result,
                "cited_passages": cited_passages,
            }
        )

        if should_rewrite(supported_label, args.correction_policy):
            corrected["generated_answer"] = NO_EVIDENCE_RESPONSE
            metadata["correction_applied"] = True
            metadata["correction_reason"] = (
                f"self-check supported={supported_label}; "
                "rewritten to insufficient evidence with no citation"
            )
        else:
            corrected["generated_answer"] = original_answer

        corrected["selfcheck_metadata"] = metadata

        append_jsonl(args.output_file, corrected)
        output_rows.append(corrected)

        action = "corrected" if metadata["correction_applied"] else "kept"
        print(
            f"[{idx}/{len(rows)}] id={row_id} "
            f"selfcheck={supported_label} action={action}"
        )

    summary = summarize(output_rows)
    write_json(args.summary_file, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
