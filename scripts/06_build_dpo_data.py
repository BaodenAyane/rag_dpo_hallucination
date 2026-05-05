#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build DPO training data from evaluated RAG generation outputs.

This script assumes you have already run 05_eval_generation.py and produced
a per-example evaluation JSONL file.

DPO construction rules:

1. Supported context + baseline abstention:
   chosen   = supported gold answer with evidence citation
   rejected = baseline abstention

2. Unsupported context + baseline non-abstention:
   chosen   = Insufficient evidence, explicitly no citation
   rejected = unsupported baseline answer

Important filtering:

- Supported examples are used conservatively.
- Unsupported examples are skipped if the baseline predicted answer itself
  appears supported by the retrieved passages.
- For yes/no predictions, if the baseline evidence is clearly copied from
  retrieved passages, the example is skipped as likely ambiguous/gold-mismatch.
- General evidence overlap is NOT used as a global skip rule, because many
  hallucinated answers cite irrelevant passages.

Example:
  python scripts/06_build_dpo_data.py \
    --eval_file outputs/baseline/base_train_eval_top10_full.jsonl \
    --output_file data/preference/dpo_train_top10_full_u2_lightclean.jsonl \
    --stats_file outputs/baseline/dpo_train_stats_top10_full_u2_lightclean.json \
    --supported_ratio 0.333 \
    --supported_mode abstention_only \
    --max_unsupported_per_supported 2.0 \
    --seed 42
"""

import argparse
import json
import os
import random
import re
import string
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


ABSTENTION_STRINGS = {
    "insufficient evidence",
    "i dont know",
    "i don't know",
    "cannot answer",
    "cant answer",
    "can't answer",
    "not enough evidence",
    "no sufficient evidence",
    "not supported",
    "not supported by provided passages",
    "not supported by the provided passages",
}

STOPWORDS_FOR_MATCH = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "at",
    "from",
}


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_for_match(text: Any) -> str:
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def content_tokens(text: Any) -> List[str]:
    tokens = normalize_for_match(text).split()
    return [t for t in tokens if t not in STOPWORDS_FOR_MATCH]


def normalize_answer_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        answers = [value]
    elif isinstance(value, list):
        answers = value
    else:
        answers = [str(value)]

    cleaned = []
    seen = set()

    for answer in answers:
        answer = str(answer).strip()
        if not answer:
            continue

        key = normalize_for_match(answer)
        if not key or key in seen:
            continue

        seen.add(key)
        cleaned.append(answer)

    return cleaned


def get_gold_answers(example: Dict[str, Any]) -> List[str]:
    answers = (
        example.get("answers")
        or example.get("gold_answers")
        or example.get("answer")
        or example.get("gold_answer")
        or []
    )
    return normalize_answer_list(answers)


def get_passage_text(passage: Dict[str, Any]) -> str:
    title = str(passage.get("title", "")).strip()
    text = str(
        passage.get("text")
        or passage.get("contents")
        or passage.get("passage")
        or ""
    ).strip()

    if title and text:
        return f"{title}: {text}"

    return text or title


def get_passage_citation_id(passage: Dict[str, Any], fallback_idx: int) -> int:
    for key in ["rank", "pid", "citation_id"]:
        if key in passage:
            try:
                return int(passage[key])
            except Exception:
                pass

    return fallback_idx


def build_context(passages: List[Dict[str, Any]]) -> str:
    lines = []

    for i, passage in enumerate(passages, start=1):
        pid = get_passage_citation_id(passage, i)
        lines.append(f"[{pid}] {get_passage_text(passage)}")

    return "\n\n".join(lines)


def build_prompt(question: str, passages: List[Dict[str, Any]]) -> str:
    context = build_context(passages)

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


def token_span_contains(text: Any, answer: Any) -> bool:
    text_tokens = normalize_for_match(text).split()
    answer_tokens = normalize_for_match(answer).split()

    if not text_tokens or not answer_tokens:
        return False

    n = len(answer_tokens)
    if n > len(text_tokens):
        return False

    for i in range(len(text_tokens) - n + 1):
        if text_tokens[i : i + n] == answer_tokens:
            return True

    return False


def relaxed_token_subset_contains(text: Any, answer: Any) -> bool:
    """
    Relaxed containment for short non-numeric answers.

    Catches examples like:
      answer: "adaptive immunity"
      text:   "Adaptive (or acquired) immunity"

      answer: "milk, meat, eggs"
      text:   "milk, meat and eggs"

    Does not aggressively match short numeric answers like "7" or "70".
    """
    answer_tokens = content_tokens(answer)
    text_tokens = set(content_tokens(text))

    if not answer_tokens or not text_tokens:
        return False

    if len(answer_tokens) == 1:
        token = answer_tokens[0]

        # Avoid over-filtering numeric answers such as "7", "70", "2001".
        if token.isdigit():
            return False

        # Avoid over-filtering very short answers.
        if len(token) <= 2:
            return False

    return all(t in text_tokens for t in answer_tokens)


def answers_overlap(a: str, b: str) -> bool:
    na = normalize_for_match(a)
    nb = normalize_for_match(b)

    if not na or not nb:
        return False

    if na == nb:
        return True

    return token_span_contains(na, nb) or token_span_contains(nb, na)


def cluster_answers(answers: List[str]) -> List[List[str]]:
    clusters: List[List[str]] = []

    for answer in answers:
        placed = False

        for cluster in clusters:
            if any(answers_overlap(answer, existing) for existing in cluster):
                cluster.append(answer)
                placed = True
                break

        if not placed:
            clusters.append([answer])

    return clusters


def representative_answer(cluster: List[str]) -> str:
    return sorted(
        cluster,
        key=lambda x: (len(normalize_for_match(x).split()), len(x)),
    )[0]


def build_answer_from_supported_gold_answers(
    supported_answers: List[str],
) -> Tuple[str, str, List[str]]:
    clusters = cluster_answers(supported_answers)
    reps = [representative_answer(cluster) for cluster in clusters]

    if len(reps) == 1:
        return reps[0], "single", reps

    return ", ".join(reps), "multi", reps


def is_pure_numeric_answer(answer: str) -> bool:
    norm = normalize_for_match(answer)
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", norm))


def find_supported_gold_answers_and_passages(
    gold_answers: List[str],
    passages: List[Dict[str, Any]],
) -> Tuple[List[str], List[int]]:
    supported_answers = []
    supporting_ids = set()

    for answer in gold_answers:
        found = False

        for i, passage in enumerate(passages, start=1):
            passage_text = get_passage_text(passage)

            if token_span_contains(passage_text, answer):
                found = True
                supporting_ids.add(get_passage_citation_id(passage, i))

        if found:
            supported_answers.append(answer)

    return supported_answers, sorted(supporting_ids)


def parse_citations(answer: Any) -> List[int]:
    if answer is None:
        return []

    citation_ids = re.findall(r"\[(\d+)\]", str(answer))
    return sorted(set(int(cid) for cid in citation_ids))


def is_abstention(example: Dict[str, Any]) -> bool:
    if "abstained" in example:
        return bool(example["abstained"])

    if "is_abstention" in example:
        return bool(example["is_abstention"])

    text = normalize_for_match(example.get("generated_answer", ""))

    return any(pattern in text for pattern in ABSTENTION_STRINGS)


def is_em_correct(example: Dict[str, Any]) -> bool:
    return float(example.get("em", 0.0)) > 0.0


def prediction_equivalent_or_contained_in_gold(example: Dict[str, Any]) -> bool:
    pred = str(example.get("pred_answer", "")).strip()

    if not pred:
        return False

    pred_norm = normalize_for_match(pred)

    if not pred_norm:
        return False

    if any(pattern == pred_norm or pattern in pred_norm for pattern in ABSTENTION_STRINGS):
        return False

    for gold in get_gold_answers(example):
        if answers_overlap(pred, gold):
            return True

    return False


def extract_evidence_text(generated_answer: Any) -> str:
    text = str(generated_answer or "").strip()

    match = re.search(r"(?is)\bevidence\s*:\s*(.*)$", text)
    if match:
        evidence = match.group(1).strip()
    else:
        evidence = text

    evidence = re.sub(r"\[\d+\]", "", evidence)
    evidence = evidence.strip().strip("\"'`").strip()

    return evidence


def evidence_appears_in_passages(
    example: Dict[str, Any],
    passages: List[Dict[str, Any]],
) -> bool:
    """
    Used only for yes/no predictions.

    If model says yes/no and its evidence is clearly copied from passages,
    the sample is likely an ambiguous or gold-mismatch case rather than
    clean unsupported hallucination.
    """
    evidence = extract_evidence_text(example.get("generated_answer", ""))

    if not evidence:
        return False

    evidence_norm = normalize_for_match(evidence)

    if not evidence_norm:
        return False

    if "insufficient evidence" in evidence_norm:
        return False

    evidence_tokens = content_tokens(evidence)

    if len(evidence_tokens) < 4:
        return False

    for passage in passages:
        passage_text = get_passage_text(passage)
        passage_norm = normalize_for_match(passage_text)

        if evidence_norm and evidence_norm in passage_norm:
            return True

        passage_tokens = set(content_tokens(passage_text))
        overlap = sum(1 for t in evidence_tokens if t in passage_tokens)
        overlap_ratio = overlap / max(1, len(evidence_tokens))

        if overlap_ratio >= 0.6:
            return True

    return False


def baseline_answer_in_passages(
    example: Dict[str, Any],
    passages: List[Dict[str, Any]],
) -> bool:
    """
    Light noise filter for unsupported DPO pairs.

    Skip only when the baseline short answer itself appears to be supported
    by retrieved passages.

    Important:
    Do NOT skip all examples whose evidence overlaps with passages, because
    hallucinated answers often quote or cite irrelevant passages.
    """
    pred = str(example.get("pred_answer", "")).strip()
    pred_norm = normalize_for_match(pred)

    if not pred_norm:
        return False

    if any(pattern == pred_norm or pattern in pred_norm for pattern in ABSTENTION_STRINGS):
        return False

    context = " ".join(get_passage_text(p) for p in passages)

    # Strict answer span match.
    # Avoid over-filtering short numeric predictions such as 7, 70, 2001.
    if not pred_norm.isdigit() and token_span_contains(context, pred):
        return True

    # Relaxed answer token matching.
    if relaxed_token_subset_contains(context, pred):
        return True

    # Yes/no answers cannot be matched by answer span.
    # If evidence is clearly passage-grounded, skip as noisy.
    if pred_norm in {"yes", "no"} and evidence_appears_in_passages(example, passages):
        return True

    return False


def is_supported(example: Dict[str, Any]) -> bool:
    if "answer_in_retrieved" in example:
        return bool(example["answer_in_retrieved"])

    for key in ["is_supported", "retrieved_answer_supported", "supported"]:
        if key in example:
            return bool(example[key])

    return False


def build_supported_chosen(
    chosen_answer: str,
    supporting_passage_ids: List[int],
) -> str:
    if supporting_passage_ids:
        citations = " ".join(f"[{pid}]" for pid in supporting_passage_ids[:2])
    else:
        citations = "[1]"

    return (
        f"Answer: {chosen_answer}\n"
        f"Evidence: The answer is supported by passage {citations}."
    )


def build_unsupported_chosen() -> str:
    return (
        "Answer: Insufficient evidence\n"
        "Evidence: The provided passages do not contain enough information to infer the answer. "
        "Do not cite any passage because none of the provided passages supports the answer."
    )


def maybe_limit_unsupported_pairs(
    supported_pairs: List[Dict[str, Any]],
    unsupported_pairs: List[Dict[str, Any]],
    max_unsupported_per_supported: Optional[float],
) -> List[Dict[str, Any]]:
    if max_unsupported_per_supported is None:
        return unsupported_pairs

    if max_unsupported_per_supported < 0:
        return unsupported_pairs

    max_unsupported = int(len(supported_pairs) * max_unsupported_per_supported)
    return unsupported_pairs[:max_unsupported]


def select_pairs_by_ratio(
    supported_pairs: List[Dict[str, Any]],
    unsupported_pairs: List[Dict[str, Any]],
    supported_ratio: float,
    max_unsupported_per_supported: Optional[float],
) -> List[Dict[str, Any]]:
    if not 0.0 <= supported_ratio <= 1.0:
        raise ValueError("--supported_ratio must be between 0 and 1.")

    if not supported_pairs and not unsupported_pairs:
        return []

    unsupported_pairs = maybe_limit_unsupported_pairs(
        supported_pairs=supported_pairs,
        unsupported_pairs=unsupported_pairs,
        max_unsupported_per_supported=max_unsupported_per_supported,
    )

    if supported_ratio == 1.0:
        final_rows = supported_pairs
        random.shuffle(final_rows)
        return final_rows

    if supported_ratio == 0.0:
        final_rows = unsupported_pairs
        random.shuffle(final_rows)
        return final_rows

    # Prefer using all supported pairs, because supported pairs are fewer.
    target_unsupported = int(
        round(len(supported_pairs) * (1.0 - supported_ratio) / supported_ratio)
    )

    if len(unsupported_pairs) >= target_unsupported:
        selected_supported = supported_pairs
        selected_unsupported = unsupported_pairs[:target_unsupported]
    else:
        selected_unsupported = unsupported_pairs

        target_supported = int(
            round(len(unsupported_pairs) * supported_ratio / (1.0 - supported_ratio))
        )
        selected_supported = supported_pairs[:target_supported]

    final_rows = selected_supported + selected_unsupported
    random.shuffle(final_rows)
    return final_rows


def build_dpo_data(
    eval_file: str,
    output_file: str,
    stats_file: Optional[str],
    supported_ratio: float,
    supported_mode: str,
    max_unsupported_per_supported: Optional[float],
    seed: int,
    filter_numeric_supported: bool,
) -> None:
    random.seed(seed)

    supported_pairs: List[Dict[str, Any]] = []
    unsupported_pairs: List[Dict[str, Any]] = []
    stats = Counter()

    for ex in read_jsonl(eval_file):
        stats["num_examples"] += 1

        ex_id = ex.get("id")
        question = str(ex.get("question", "")).strip()
        passages = ex.get("retrieved_passages") or ex.get("passages") or []
        rejected = str(ex.get("generated_answer", "")).strip()

        if not question:
            stats["skipped_missing_question"] += 1
            continue

        if not passages:
            stats["skipped_missing_passages"] += 1
            continue

        if not rejected:
            stats["skipped_missing_rejected"] += 1
            continue

        prompt = build_prompt(question, passages)

        supported = is_supported(ex)
        abstained = is_abstention(ex)
        em_correct = is_em_correct(ex)
        equivalent_or_contained = prediction_equivalent_or_contained_in_gold(ex)
        baseline_citation_ids = parse_citations(rejected)

        if supported:
            stats["supported_examples"] += 1

            gold_answers = get_gold_answers(ex)

            if not gold_answers:
                stats["skipped_missing_gold_answer"] += 1
                continue

            supported_gold_answers, supporting_passage_ids = (
                find_supported_gold_answers_and_passages(gold_answers, passages)
            )

            if not supported_gold_answers or not supporting_passage_ids:
                stats["supported_missing_or_filtered_support"] += 1
                continue

            if filter_numeric_supported and all(
                is_pure_numeric_answer(answer) for answer in supported_gold_answers
            ):
                stats["supported_numeric_filtered"] += 1
                continue

            chosen_answer, answer_set_type, chosen_answer_list = (
                build_answer_from_supported_gold_answers(supported_gold_answers)
            )

            if answer_set_type == "multi":
                stats["supported_multi_answer_candidates"] += 1

            should_make_supported_pair = False

            if supported_mode == "abstention_only":
                should_make_supported_pair = abstained
                if not abstained:
                    stats["supported_non_abstain_skipped"] += 1
            elif supported_mode == "wrong_or_abstain":
                should_make_supported_pair = abstained or not em_correct
                if em_correct and not abstained:
                    stats["supported_already_correct"] += 1
            else:
                raise ValueError(
                    "--supported_mode must be either 'abstention_only' or 'wrong_or_abstain'."
                )

            if not should_make_supported_pair:
                continue

            chosen = build_supported_chosen(
                chosen_answer=chosen_answer,
                supporting_passage_ids=supporting_passage_ids,
            )

            supported_pairs.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "type": "supported_answer",
                    "id": ex_id,
                    "question": question,
                    "chosen_answer": chosen_answer,
                    "gold_answers": gold_answers,
                    "supported_gold_answers": chosen_answer_list,
                    "answer_set_type": answer_set_type,
                    "supporting_passage_ids": supporting_passage_ids,
                    "baseline_pred_answer": ex.get("pred_answer"),
                    "baseline_abstained": abstained,
                    "baseline_correct": em_correct,
                    "baseline_em_correct": em_correct,
                    "answer_in_retrieved": supported,
                    "baseline_citation_ids": baseline_citation_ids,
                }
            )
            stats["supported_pairs"] += 1

        else:
            stats["unsupported_examples"] += 1

            if em_correct:
                stats["unsupported_but_correct_skipped"] += 1
                continue

            if equivalent_or_contained:
                stats["non_em_but_equivalent_or_contained"] += 1
                continue

            if baseline_answer_in_passages(ex, passages):
                stats["unsupported_baseline_answer_in_passages_skipped"] += 1
                continue

            if abstained:
                stats["unsupported_already_abstained"] += 1
                continue

            chosen = build_unsupported_chosen()

            unsupported_pairs.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "type": "unsupported_abstention",
                    "id": ex_id,
                    "question": question,
                    "gold_answers": get_gold_answers(ex),
                    "answer_set_type": "single",
                    "baseline_pred_answer": ex.get("pred_answer"),
                    "baseline_abstained": abstained,
                    "baseline_correct": em_correct,
                    "baseline_em_correct": em_correct,
                    "answer_in_retrieved": supported,
                    "baseline_citation_ids": baseline_citation_ids,
                    "baseline_answer_in_passages": False,
                }
            )
            stats["unsupported_pairs"] += 1

    random.shuffle(supported_pairs)
    random.shuffle(unsupported_pairs)

    dpo_rows = select_pairs_by_ratio(
        supported_pairs=supported_pairs,
        unsupported_pairs=unsupported_pairs,
        supported_ratio=supported_ratio,
        max_unsupported_per_supported=max_unsupported_per_supported,
    )

    write_jsonl(output_file, dpo_rows)

    stats["available_supported_pairs"] = len(supported_pairs)
    stats["available_unsupported_pairs"] = len(unsupported_pairs)
    stats["final_dpo_examples"] = len(dpo_rows)
    stats["final_supported_pairs"] = sum(
        x["type"] == "supported_answer" for x in dpo_rows
    )
    stats["final_unsupported_pairs"] = sum(
        x["type"] == "unsupported_abstention" for x in dpo_rows
    )
    stats["supported_ratio_requested"] = supported_ratio
    stats["max_unsupported_per_supported"] = max_unsupported_per_supported
    stats["supported_mode"] = supported_mode
    stats["filter_numeric_supported"] = filter_numeric_supported

    if stats_file:
        stats_dir = os.path.dirname(stats_file)
        if stats_dir:
            os.makedirs(stats_dir, exist_ok=True)

        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(dict(stats), f, ensure_ascii=False, indent=2)

    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--stats_file", default=None)
    parser.add_argument(
        "--supported_ratio",
        type=float,
        default=0.5,
        help=(
            "Target ratio of supported-answer pairs in final DPO data. "
            "For example, 0.333 gives roughly supported:unsupported = 1:2."
        ),
    )
    parser.add_argument(
        "--supported_mode",
        type=str,
        default="abstention_only",
        choices=["abstention_only", "wrong_or_abstain"],
        help=(
            "How to build supported-answer pairs. "
            "'abstention_only' only uses supported examples where the baseline abstained. "
            "'wrong_or_abstain' also uses supported examples where the baseline was EM-wrong."
        ),
    )
    parser.add_argument(
        "--max_unsupported_per_supported",
        type=float,
        default=None,
        help=(
            "Optional cap on unsupported pairs per supported pair. "
            "For example, 2.0 means at most 2 unsupported pairs for each supported pair."
        ),
    )
    parser.add_argument(
        "--no_filter_numeric_supported",
        action="store_true",
        help="Disable filtering of short purely numeric supported answers.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    build_dpo_data(
        eval_file=args.eval_file,
        output_file=args.output_file,
        stats_file=args.stats_file,
        supported_ratio=args.supported_ratio,
        supported_mode=args.supported_mode,
        max_unsupported_per_supported=args.max_unsupported_per_supported,
        seed=args.seed,
        filter_numeric_supported=not args.no_filter_numeric_supported,
    )


if __name__ == "__main__":
    main()
    