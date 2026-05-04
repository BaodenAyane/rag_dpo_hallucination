#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build DPO training data from evaluated RAG generation outputs.

This script assumes you have already run 05_eval_generation.py and produced
a per-example evaluation JSONL file.

Important:
  The per-example eval file must keep "retrieved_passages".
  In 05_eval_generation.py, make sure evaluate_record() returns:
    "retrieved_passages": retrieved_passages

Example:
  python scripts/06_build_dpo_data.py \
    --eval_file outputs/baseline/base_val_eval_top10_1_shards.jsonl \
    --output_file data/preference/dpo_val_top10_1_shards.jsonl \
    --stats_file outputs/baseline/dpo_val_stats_top10_1_shards.json \
    --supported_ratio 0.6 \
    --seed 42

Input JSONL should contain fields similar to:
{
  "id": "0",
  "question": "when was the last time anyone was on the moon",
  "answers": ["14 December 1972 UTC", "December 1972"],
  "retrieved_passages": [
    {"pid": 1, "title": "...", "text": "..."},
    ...
  ],
  "generated_answer": "Answer: December 1972\\nEvidence: ... [5]",
  "pred_answer": "December 1972",
  "em": 1.0,
  "f1": 1.0,
  "has_citation": true,
  "citation_ids": [5],
  "abstained": false,
  "answer_in_retrieved": true
}

Output JSONL format:
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "type": "supported_answer" or "unsupported_abstention"
}

DPO construction rules:
  1. answer_in_retrieved = true and baseline is wrong or abstained:
       chosen   = gold answer with citation
       rejected = baseline generation

  2. answer_in_retrieved = false and baseline is wrong and did not abstain:
       chosen   = Insufficient evidence
       rejected = baseline generation

  3. answer_in_retrieved = false but baseline is correct:
       skip
       This avoids teaching the model to reject correct answers due to noisy support labels.
"""

import argparse
import json
import os
import random
from collections import Counter


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_gold_answer(example):
    answers = (
        example.get("answers")
        or example.get("gold_answers")
        or example.get("answer")
        or example.get("gold_answer")
    )

    if isinstance(answers, list):
        answers = [str(a).strip() for a in answers if str(a).strip()]
        if not answers:
            return ""
        return sorted(answers, key=lambda x: (len(x.split()), len(x)))[0]

    if isinstance(answers, str):
        return answers.strip()

    return ""


def get_passage_text(passage):
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


def build_context(passages):
    lines = []
    for i, p in enumerate(passages, start=1):
        pid = p.get("pid", p.get("id", i))
        lines.append(f"[{pid}] {get_passage_text(p)}")
    return "\n\n".join(lines)


def build_prompt(question, passages):
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


def get_supporting_ids(example):
    """
    Prefer explicit supporting_passage_ids if available.
    Otherwise use citation_ids as a weak fallback.
    """
    supporting_ids = example.get("supporting_passage_ids")

    if isinstance(supporting_ids, list) and supporting_ids:
        return supporting_ids

    citation_ids = example.get("citation_ids")
    if isinstance(citation_ids, list) and citation_ids:
        return citation_ids

    return []


def build_supported_chosen(gold_answer, supporting_passage_ids):
    if supporting_passage_ids:
        citation = f"[{supporting_passage_ids[0]}]"
    else:
        citation = "[1]"

    return (
        f"Answer: {gold_answer}\n"
        f"Evidence: The answer is supported by passage {citation}."
    )


def build_unsupported_chosen():
    return (
        "Answer: Insufficient evidence\n"
        "Evidence: The provided passages do not contain enough information to infer the answer."
    )


def is_correct(example):
    """
    Use the field produced by 05_eval_generation.py:
      em: 1.0 if prediction exactly matches any gold answer.
    """
    return float(example.get("em", 0.0)) > 0.0


def is_abstention(example):
    """
    Use the field produced by 05_eval_generation.py:
      abstained: bool
    Fallback to string matching if the field is missing.
    """
    if "abstained" in example:
        return bool(example["abstained"])

    if "is_abstention" in example:
        return bool(example["is_abstention"])

    output = str(example.get("generated_answer", "")).lower()
    return (
        "insufficient evidence" in output
        or "not enough evidence" in output
        or "cannot answer" in output
    )


def is_supported(example):
    """
    Use the field produced by 05_eval_generation.py:
      answer_in_retrieved: bool
    This means at least one gold answer string appears in retrieved passages.
    """
    if "answer_in_retrieved" in example:
        return bool(example["answer_in_retrieved"])

    for key in ["is_supported", "retrieved_answer_supported", "supported"]:
        if key in example:
            return bool(example[key])

    return False


def build_dpo_data(eval_file, output_file, stats_file, supported_ratio, seed):
    random.seed(seed)

    supported_pairs = []
    unsupported_pairs = []
    stats = Counter()

    for ex in read_jsonl(eval_file):
        stats["num_examples"] += 1

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
        correct = is_correct(ex)

        if supported:
            stats["supported_examples"] += 1

            gold_answer = get_gold_answer(ex)
            supporting_ids = get_supporting_ids(ex)

            if not gold_answer:
                stats["skipped_missing_gold_answer"] += 1
                continue

            # DPO case 1:
            # Evidence exists, but baseline abstained or answered incorrectly.
            if abstained or not correct:
                chosen = build_supported_chosen(gold_answer, supporting_ids)
                supported_pairs.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "type": "supported_answer",
                        "id": ex.get("id"),
                        "question": question,
                        "gold_answer": gold_answer,
                        "baseline_pred_answer": ex.get("pred_answer"),
                        "baseline_abstained": abstained,
                        "baseline_correct": correct,
                        "answer_in_retrieved": supported,
                        "citation_ids": ex.get("citation_ids", []),
                    }
                )
                stats["supported_pairs"] += 1
            else:
                stats["supported_already_correct"] += 1

        else:
            stats["unsupported_examples"] += 1

            # Important:
            # If the model answered correctly even though answer_in_retrieved is false,
            # skip it. The support label may be noisy, or the model may have used
            # parametric knowledge. Do not train the model to reject correct answers.
            if correct:
                stats["unsupported_but_correct_skipped"] += 1
                continue

            # DPO case 2:
            # No retrieved evidence and baseline still answered.
            if not abstained:
                chosen = build_unsupported_chosen()
                unsupported_pairs.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "type": "unsupported_abstention",
                        "id": ex.get("id"),
                        "question": question,
                        "baseline_pred_answer": ex.get("pred_answer"),
                        "baseline_abstained": abstained,
                        "baseline_correct": correct,
                        "answer_in_retrieved": supported,
                        "citation_ids": ex.get("citation_ids", []),
                    }
                )
                stats["unsupported_pairs"] += 1
            else:
                stats["unsupported_already_abstained"] += 1

    random.shuffle(supported_pairs)
    random.shuffle(unsupported_pairs)

    total = len(supported_pairs) + len(unsupported_pairs)

    if total == 0:
        dpo_rows = []
    else:
        target_supported = int(total * supported_ratio)
        target_unsupported = total - target_supported

        selected_supported = supported_pairs[:target_supported]
        selected_unsupported = unsupported_pairs[:target_unsupported]

        # Fill remaining slots if one side does not have enough examples.
        remaining = total - len(selected_supported) - len(selected_unsupported)
        extras = supported_pairs[target_supported:] + unsupported_pairs[target_unsupported:]
        random.shuffle(extras)

        dpo_rows = selected_supported + selected_unsupported + extras[:remaining]
        random.shuffle(dpo_rows)

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

    if stats_file:
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(dict(stats), f, ensure_ascii=False, indent=2)

    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--stats_file", default=None)
    parser.add_argument(
        "--supported_ratio",
        type=float,
        default=0.6,
        help="Target ratio of supported-answer pairs in final DPO data.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not 0.0 <= args.supported_ratio <= 1.0:
        raise ValueError("--supported_ratio must be between 0 and 1.")

    build_dpo_data(
        eval_file=args.eval_file,
        output_file=args.output_file,
        stats_file=args.stats_file,
        supported_ratio=args.supported_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
