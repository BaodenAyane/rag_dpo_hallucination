"""
Evaluate baseline RAG generations.

This evaluation is designed for RAG behavior, not only QA exact match.

Core logic:
  1. If retrieved passages contain/support the gold answer:
       model should answer correctly.
  2. If retrieved passages do NOT contain/support the gold answer:
       model should abstain / say insufficient evidence.

Example:
  python scripts/05_eval_generation.py \
    --generation_file data/generation/base_validation_outputs_top10_full.jsonl \
    --output_file outputs/baseline/base_validation_metrics_top10_full.json \
    --per_example_output outputs/baseline/base_validation_eval_top10_full.jsonl
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
    """
    Check whether retrieved passages contain any gold answer.

    This is a weak proxy for evidence sufficiency:
      True  -> retrieved context likely contains enough evidence.
      False -> retrieved context likely lacks answer evidence.

    Later you can replace this with an LLM judge or NLI-based support checker.
    """
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
        "i do not know",
        "insufficient evidence",
        "provided evidence is insufficient",
        "the provided evidence is insufficient",
        "not enough evidence",
        "cannot answer",
        "can not answer",
        "cant answer",
        "not supported by provided passages",
        "not supported by the provided passages",
        "no sufficient evidence",
        "evidence is insufficient",
        "the context does not provide",
        "the provided context does not provide",
        "the passages do not provide",
        "the provided passages do not provide",
        "the context does not contain",
        "the provided context does not contain",
        "the passages do not contain",
        "the provided passages do not contain",
        "based on the provided evidence i dont know",
        "based on the provided evidence i do not know",
        "not enough information",
        "there is not enough information",
        "there isnt enough information",
        "there is insufficient information",
    ]

    return any(pattern in normalized for pattern in abstention_patterns)


def answer_matches_gold(prediction: str, gold_answers: List[str]) -> bool:
    """
    Check whether the predicted answer matches any gold answer.

    We use both exact match and token-span containment because RAG generations
    may be short sentences rather than pure answer strings.
    """
    if exact_match(prediction, gold_answers) == 1.0:
        return True

    return contains_gold_answer(prediction, gold_answers)


def classify_response(
    evidence_sufficient: bool,
    abstained: bool,
    answer_correct: bool,
) -> str:
    """
    Classify the model behavior under the retrieved-evidence condition.

    Desired behavior:
      - If evidence is sufficient: answer correctly.
      - If evidence is insufficient: abstain.

    Labels:
      supported_answer:
        Evidence is sufficient and the model answers correctly.

      unsupported_answer:
        Evidence is sufficient, but the model gives an incorrect or unsupported answer.

      over_refusal:
        Evidence is sufficient, but the model refuses to answer.

      correct_refusal:
        Evidence is insufficient, and the model refuses to answer.

      overconfident_answer:
        Evidence is insufficient, but the model still gives a concrete answer.
    """
    if evidence_sufficient:
        if abstained:
            return "over_refusal"
        if answer_correct:
            return "supported_answer"
        return "unsupported_answer"

    if abstained:
        return "correct_refusal"

    return "overconfident_answer"


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
    """Evaluate one generated example under RAG behavior metrics."""
    raw_answer = record.get("generated_answer", "")
    pred_answer = extract_prediction_answer(raw_answer)

    gold_answers = record.get("answers", [])
    retrieved_passages = record.get("retrieved_passages", [])

    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]

    gold_answers = [str(x) for x in gold_answers if str(x).strip()]

    # 1. Evidence status.
    # In this first version, evidence_sufficient means the retrieved passages
    # contain the gold answer string.
    evidence_sufficient = retrieved_answer_recall(retrieved_passages, gold_answers)
    evidence_status = "sufficient" if evidence_sufficient else "insufficient"

    # 2. Abstention.
    abstained = is_abstention(raw_answer)

    # 3. QA correctness.
    # EM/F1 are meaningful mainly for evidence_sufficient=True examples.
    if abstained:
        em = 0.0
        f1 = 0.0
        answer_correct = False
    else:
        em = exact_match(pred_answer, gold_answers)
        f1 = max_f1_score(pred_answer, gold_answers)
        answer_correct = answer_matches_gold(pred_answer, gold_answers)

    # 4. Citation parsing.
    citation_ids = parse_citations(raw_answer)
    has_citation = len(citation_ids) > 0

    valid_passage_ids = get_valid_passage_ids(retrieved_passages)
    valid_citation_ids = [cid for cid in citation_ids if cid in valid_passage_ids]
    invalid_citation_ids = [cid for cid in citation_ids if cid not in valid_passage_ids]
    has_valid_citation = len(valid_citation_ids) > 0

    # 5. RAG behavior classification.
    response_type = classify_response(
        evidence_sufficient=evidence_sufficient,
        abstained=abstained,
        answer_correct=answer_correct,
    )

    rag_behavior_correct = response_type in {
        "supported_answer",
        "correct_refusal",
    }

    policy_error = response_type in {
        "unsupported_answer",
        "over_refusal",
        "overconfident_answer",
    }

    bad_answer_behavior = response_type in {
        "unsupported_answer",
        "overconfident_answer",
    }

    return {
        "id": record.get("id"),
        "question": record.get("question"),
        "answers": gold_answers,

        # Reusable fields for later scripts.
        "prompt": record.get("prompt"),
        "retrieved_passages": retrieved_passages,

        # Generation.
        "generated_answer": raw_answer,
        "pred_answer": pred_answer,

        # Evidence status.
        "evidence_sufficient": evidence_sufficient,
        "evidence_status": evidence_status,

        # Backward compatibility with your previous field.
        "answer_in_retrieved": evidence_sufficient,

        # Abstention.
        "abstained": abstained,

        # QA metrics.
        # Use these mainly for evidence_sufficient=True examples.
        "em": em,
        "f1": f1,
        "answer_correct": answer_correct,

        # Citation metrics.
        "has_citation": has_citation,
        "citation_ids": citation_ids,
        "valid_citation_ids": valid_citation_ids,
        "invalid_citation_ids": invalid_citation_ids,
        "has_valid_citation": has_valid_citation,

        # Main RAG behavior labels.
        "response_type": response_type,
        "rag_behavior_correct": rag_behavior_correct,
        "policy_error": policy_error,
        "bad_answer_behavior": bad_answer_behavior,

        # Convenience binary fields.
        "is_supported_answer": response_type == "supported_answer",
        "is_unsupported_answer": response_type == "unsupported_answer",
        "is_correct_refusal": response_type == "correct_refusal",
        "is_over_refusal": response_type == "over_refusal",
        "is_overconfident_answer": response_type == "overconfident_answer",

        # Generation metadata.
        "model_name": record.get("model_name"),
        "top_k": record.get("top_k"),
        "max_new_tokens": record.get("max_new_tokens"),
        "max_input_length": record.get("max_input_length"),
    }


def mean(values: List[float]) -> float:
    """Safe mean."""
    return sum(values) / len(values) if values else 0.0


def summarize(eval_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate RAG behavior metrics."""
    n = len(eval_records)

    if n == 0:
        return {}

    sufficient = [r for r in eval_records if r["evidence_sufficient"]]
    insufficient = [r for r in eval_records if not r["evidence_sufficient"]]

    response_type_counts = Counter(r["response_type"] for r in eval_records)

    def rate(label: str, records: List[Dict[str, Any]]) -> float:
        if not records:
            return 0.0
        return mean([float(r["response_type"] == label) for r in records])

    def bool_rate(field: str, records: List[Dict[str, Any]]) -> float:
        if not records:
            return 0.0
        return mean([float(r[field]) for r in records])

    metrics = {
        "num_examples": n,

        # ============================================================
        # Main metrics.
        # These should be your primary reported metrics.
        # ============================================================

        # Correct RAG behavior:
        #   sufficient evidence -> supported answer
        #   insufficient evidence -> correct refusal
        "rag_behavior_accuracy": bool_rate("rag_behavior_correct", eval_records),

        # Any policy-level error:
        #   unsupported_answer, over_refusal, or overconfident_answer
        "policy_error_rate": bool_rate("policy_error", eval_records),

        # Hallucination-style bad answering behavior:
        #   1. evidence sufficient but answer is wrong/unsupported
        #   2. evidence insufficient but model still answers confidently
        # This excludes over-refusal.
        "bad_answer_behavior_rate": bool_rate("bad_answer_behavior", eval_records),

        # ============================================================
        # Evidence split.
        # ============================================================

        "retrieved_answer_recall": bool_rate("evidence_sufficient", eval_records),
        "evidence_sufficient_num_examples": len(sufficient),
        "evidence_insufficient_num_examples": len(insufficient),

        # ============================================================
        # Response type distribution.
        # ============================================================

        "supported_answer_count": response_type_counts.get("supported_answer", 0),
        "unsupported_answer_count": response_type_counts.get("unsupported_answer", 0),
        "correct_refusal_count": response_type_counts.get("correct_refusal", 0),
        "over_refusal_count": response_type_counts.get("over_refusal", 0),
        "overconfident_answer_count": response_type_counts.get(
            "overconfident_answer", 0
        ),

        "supported_answer_rate": rate("supported_answer", eval_records),
        "unsupported_answer_rate": rate("unsupported_answer", eval_records),
        "correct_refusal_rate": rate("correct_refusal", eval_records),
        "over_refusal_rate": rate("over_refusal", eval_records),
        "overconfident_answer_rate": rate("overconfident_answer", eval_records),

        # ============================================================
        # Sufficient-evidence subset.
        # EM/F1 are meaningful here.
        # ============================================================

        "sufficient_exact_match": mean([r["em"] for r in sufficient]),
        "sufficient_f1": mean([r["f1"] for r in sufficient]),
        "sufficient_answer_accuracy": bool_rate("answer_correct", sufficient),

        "sufficient_supported_answer_rate": rate("supported_answer", sufficient),
        "sufficient_unsupported_answer_rate": rate("unsupported_answer", sufficient),
        "sufficient_over_refusal_rate": rate("over_refusal", sufficient),

        "sufficient_abstention_rate": bool_rate("abstained", sufficient),
        "sufficient_citation_rate": bool_rate("has_citation", sufficient),
        "sufficient_valid_citation_rate": bool_rate(
            "has_valid_citation", sufficient
        ),

        # ============================================================
        # Insufficient-evidence subset.
        # EM/F1 are NOT the main metrics here.
        # Desired behavior is refusal.
        # ============================================================

        "insufficient_correct_refusal_rate": rate("correct_refusal", insufficient),
        "insufficient_overconfident_answer_rate": rate(
            "overconfident_answer", insufficient
        ),
        "insufficient_abstention_rate": bool_rate("abstained", insufficient),

        "insufficient_citation_rate": bool_rate("has_citation", insufficient),
        "insufficient_valid_citation_rate": bool_rate(
            "has_valid_citation", insufficient
        ),

        # ============================================================
        # Diagnostic-only overall QA metrics.
        # Do not use these as primary metrics because insufficient-evidence
        # examples should not be rewarded for matching the gold answer.
        # ============================================================

        "diagnostic_overall_exact_match": mean([r["em"] for r in eval_records]),
        "diagnostic_overall_f1": mean([r["f1"] for r in eval_records]),
        "diagnostic_overall_answer_accuracy": bool_rate(
            "answer_correct", eval_records
        ),

        # ============================================================
        # Overall citation and abstention diagnostics.
        # ============================================================

        "overall_citation_rate": bool_rate("has_citation", eval_records),
        "overall_valid_citation_rate": bool_rate(
            "has_valid_citation", eval_records
        ),
        "overall_abstention_rate": bool_rate("abstained", eval_records),
    }

    return metrics


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
