"""
Evaluate retrieval recall for BM25 retrieval results.

The metric is weak answer recall:
a retrieval hit means any gold answer string appears as a token span
in the top-k retrieved passages.

Example:
  python scripts/03_eval_retrieval.py \
    --retrieval_file data/retrieval/nq_validation_bm25_top10_50_shards.jsonl \
    --output_file outputs/retrieval/wiki_dpr_50_shards_metrics.json \
    --ks 1 5 10 20 50
"""

import argparse
import json
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterator, List
from collections import Counter


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream records from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_text(text: str) -> str:
    """Normalize text for token-span answer matching."""
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def normalize_answers(answers: Any) -> List[str]:
    """Normalize answer fields to a list of strings."""
    if isinstance(answers, str):
        return [answers]

    if isinstance(answers, list):
        return [str(answer) for answer in answers if str(answer).strip()]

    return []


def contains_answer_span(text: str, answer: str) -> bool:
    """Check whether answer appears as a contiguous token span in text."""
    text_tokens = normalize_text(text).split()
    answer_tokens = normalize_text(answer).split()

    if not text_tokens or not answer_tokens:
        return False

    n = len(answer_tokens)
    if n > len(text_tokens):
        return False

    for i in range(len(text_tokens) - n + 1):
        if text_tokens[i : i + n] == answer_tokens:
            return True

    return False


def answer_in_passages(
    answers: List[str],
    passages: List[Dict[str, Any]],
    k: int,
) -> bool:
    """Check whether any gold answer appears in the top-k passages."""
    topk_passages = passages[:k]

    context = " ".join(
        f"{passage.get('title', '')} {passage.get('text', '')}"
        for passage in topk_passages
    )

    for answer in answers:
        if contains_answer_span(context, answer):
            return True

    return False


def evaluate_recall_at_ks(
    records: List[Dict[str, Any]],
    ks: List[int],
) -> Dict[str, Any]:
    """Compute weak answer recall@k for multiple k values."""
    total = len(records)
    metrics: Dict[str, Any] = {
        "num_examples": total,
    }

    retrieved_counts = [len(r.get("retrieved_passages", [])) for r in records]
    metrics["avg_retrieved_passages"] = (
        sum(retrieved_counts) / total if total > 0 else 0.0
    )
    metrics["empty_retrievals"] = sum(1 for c in retrieved_counts if c == 0)

    for k in ks:
        hits = 0

        for record in records:
            answers = normalize_answers(record.get("answers", []))
            passages = record.get("retrieved_passages", [])

            if answer_in_passages(answers, passages, k):
                hits += 1

        recall = hits / total if total > 0 else 0.0
        metrics[f"recall_at_{k}"] = recall
        metrics[f"hits_at_{k}"] = hits

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, default=None)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args()

    records = list(read_jsonl(args.retrieval_file))
    metrics = evaluate_recall_at_ks(records, args.ks)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with args.output_file.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"Saved retrieval metrics to: {args.output_file}")


if __name__ == "__main__":
    main()
