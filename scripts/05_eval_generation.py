"""
Evaluate baseline RAG generations.

Example:
  python scripts/05_eval_generation.py \
    --generation_file data/generation/base_val_outputs_top10_full.jsonl \
    --output_file outputs/baseline/base_val_metrics_top10_full.json \
    --per_example_output outputs/baseline/base_val_eval_top10_full.jsonl
"""

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    """Write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_answer(text: str) -> str:
    """Normalize text for EM/F1 evaluation."""
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def extract_prediction_answer(generated: str) -> str:
    """
    Extract the short answer from a RAG generation.

    Expected preferred output format:
      Answer: <short answer>
      Evidence: <brief evidence with citations>
    """
    if generated is None:
        return ""

    text = str(generated).strip()

    answer_match = re.search(
        r"(?is)\banswer\s*:\s*(.*?)(?:\n\s*(?:evidence|citation|citations|support|explanation|reasoning)\s*:|$)",
        text,
    )

    if answer_match:
        text = answer_match.group(1).strip()
    else:
        text = re.split(
            r"(?is)\n\s*(?:evidence|citation|citations|support|explanation|reasoning)\s*:",
            text,
            maxsplit=1,
        )[0].strip()

    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"(?i)^answer\s*:\s*", "", text).strip()
    text = re.sub(r"(?i)^the answer is\s+", "", text).strip()
    text = re.sub(r"(?i)^it is\s+", "", text).strip()
    text = text.splitlines()[0].strip()
    text = text.strip().strip("\"'`").strip()
    text = text.strip(" .")

    return text


def exact_match(prediction: str, gold_answers: List[str]) -> float:
    """Return 1 if prediction exactly matches any gold answer after normalization."""
    pred = normalize_answer(prediction)

    for gold in gold_answers:
        if pred == normalize_answer(gold):
            return 1.0

    return 0.0


def f1_score(prediction: str, gold_answer: str) -> float:
    """Compute token-level F1 against one gold answer."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def max_f1_score(prediction: str, gold_answers: List[str]) -> float:
    """Return the maximum F1 over all gold answers."""
    if not gold_answers:
        return 0.0

    return max(f1_score(prediction, gold) for gold in gold_answers)


def contains_gold_answer(text: str, gold_answers: List[str]) -> bool:
    """
    Check whether any gold answer appears as a token span in a text block.
    This avoids false positives such as answer "us" matching inside "thus".
    """
    normalized_text = normalize_answer(text)
    text_tokens = normalized_text.split()

    if not text_tokens:
        return False

    for answer in gold_answers:
        normalized_answer = normalize_answer(answer)
        answer_tokens = normalized_answer.split()

        if not answer_tokens:
            continue

        n = len(answer_tokens)
        if n > len(text_tokens):
            continue

        for i in range(len(text_tokens) - n + 1):
            if text_tokens[i : i + n] == answer_tokens:
                return True

    return False


def retrieved_answer_recall(
    retrieved_passages: List[Dict[str, Any]],
    gold_answers: List[str],
) -> bool:
    """Check whether retrieved passages contain any gold answer."""
    context = " ".join(
        f"{p.get('title', '')} {p.get('text', '')}"
        for p in retrieved_passages
    )
    return contains_gold_answer(context, gold_answers)


def parse_citations(answer: str) -> List[int]:
    """Extract citation IDs like [1], [2], [10] from generated answer."""
    if answer is None:
        return []

    citation_ids = re.findall(r"\[(\d+)\]", str(answer))
    return sorted(set(int(cid) for cid in citation_ids))


def is_abstention(answer: str) -> bool:
    """Detect whether the model abstained from answering."""
    normalized = normalize_answer(answer)

    abstention_patterns = [
        "i dont know",
        "insufficient evidence",
        "provided evidence is insufficient",
        "not enough evidence",
        "cannot answer",
        "cant answer",
        "not supported by provided passages",
        "not supported by the provided passages",
        "no sufficient evidence",
        "evidence is insufficient",
    ]

    return any(pattern in normalized for pattern in abstention_patterns)


def get_valid_passage_ids(retrieved_passages: List[Dict[str, Any]]) -> set:
    """
    Get valid citation IDs from retrieved passages.

    In 04_generate_baseline.py, passages are formatted using passage["rank"]
    when available, otherwise their 1-based position.
    """
    valid_ids = set()

    for i, passage in enumerate(retrieved_passages, start=1):
        try:
            valid_ids.add(int(passage.get("rank", i)))
        except Exception:
            valid_ids.add(i)

    return valid_ids


def evaluate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one generated example."""
    raw_answer = record.get("generated_answer", "")
    pred_answer = extract_prediction_answer(raw_answer)

    gold_answers = record.get("answers", [])
    retrieved_passages = record.get("retrieved_passages", [])

    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]

    gold_answers = [str(x) for x in gold_answers if str(x).strip()]

    abstained = is_abstention(raw_answer)

    if abstained:
        em = 0.0
        f1 = 0.0
    else:
        em = exact_match(pred_answer, gold_answers)
        f1 = max_f1_score(pred_answer, gold_answers)

    citation_ids = parse_citations(raw_answer)
    has_citation = len(citation_ids) > 0

    valid_passage_ids = get_valid_passage_ids(retrieved_passages)
    valid_citation_ids = [cid for cid in citation_ids if cid in valid_passage_ids]
    invalid_citation_ids = [cid for cid in citation_ids if cid not in valid_passage_ids]
    has_valid_citation = len(valid_citation_ids) > 0

    answer_in_retrieved = retrieved_answer_recall(retrieved_passages, gold_answers)

    return {
        "id": record.get("id"),
        "question": record.get("question"),
        "answers": gold_answers,

        # Keep original prompt and retrieved passages so later scripts can reuse them.
        "prompt": record.get("prompt"),
        "retrieved_passages": retrieved_passages,

        "generated_answer": raw_answer,
        "pred_answer": pred_answer,
        "em": em,
        "f1": f1,

        "has_citation": has_citation,
        "citation_ids": citation_ids,
        "valid_citation_ids": valid_citation_ids,
        "invalid_citation_ids": invalid_citation_ids,
        "has_valid_citation": has_valid_citation,

        "abstained": abstained,
        "answer_in_retrieved": answer_in_retrieved,

        # Generation metadata for reproducibility.
        "model_name": record.get("model_name"),
        "top_k": record.get("top_k"),
        "max_new_tokens": record.get("max_new_tokens"),
        "max_input_length": record.get("max_input_length"),
    }


def mean(values: List[float]) -> float:
    """Safe mean."""
    return sum(values) / len(values) if values else 0.0


def summarize(eval_records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate per-example evaluation results."""
    n = len(eval_records)

    if n == 0:
        return {}

    supported = [r for r in eval_records if r["answer_in_retrieved"]]
    unsupported = [r for r in eval_records if not r["answer_in_retrieved"]]

    return {
        "num_examples": n,

        "exact_match": mean([r["em"] for r in eval_records]),
        "f1": mean([r["f1"] for r in eval_records]),
        "citation_rate": mean([float(r["has_citation"]) for r in eval_records]),
        "valid_citation_rate": mean(
            [float(r["has_valid_citation"]) for r in eval_records]
        ),
        "abstention_rate": mean([float(r["abstained"]) for r in eval_records]),
        "retrieved_answer_recall": mean(
            [float(r["answer_in_retrieved"]) for r in eval_records]
        ),

        "supported_num_examples": len(supported),
        "supported_exact_match": mean([r["em"] for r in supported]),
        "supported_f1": mean([r["f1"] for r in supported]),
        "supported_citation_rate": mean(
            [float(r["has_citation"]) for r in supported]
        ),
        "supported_valid_citation_rate": mean(
            [float(r["has_valid_citation"]) for r in supported]
        ),
        "supported_abstention_rate": mean(
            [float(r["abstained"]) for r in supported]
        ),

        "unsupported_num_examples": len(unsupported),
        "unsupported_exact_match": mean([r["em"] for r in unsupported]),
        "unsupported_f1": mean([r["f1"] for r in unsupported]),
        "unsupported_citation_rate": mean(
            [float(r["has_citation"]) for r in unsupported]
        ),
        "unsupported_valid_citation_rate": mean(
            [float(r["has_valid_citation"]) for r in unsupported]
        ),
        "unsupported_abstention_rate": mean(
            [float(r["abstained"]) for r in unsupported]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--per_example_output", type=Path, default=None)
    args = parser.parse_args()

    records = read_jsonl(args.generation_file)
    eval_records = [evaluate_record(record) for record in records]
    metrics = summarize(eval_records)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if args.per_example_output is not None:
        write_jsonl(eval_records, args.per_example_output)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
