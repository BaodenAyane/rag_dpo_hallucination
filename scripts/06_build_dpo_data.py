"""
Build DPO training data from evaluated RAG generation outputs.

Recommended first-run command:
  python scripts/06_build_dpo_data.py \
    --eval_file outputs/baseline/base_train_eval_top10_full.jsonl \
    --output_file data/preference/dpo_train_top10_full.jsonl \
    --stats_file outputs/baseline/dpo_train_stats_top10_full.json \
    --supported_ratio 0.5 \
    --supported_mode abstention_only \
    --seed 42

Design principles:
  1. Prefer high-quality preference pairs over quantity.
  2. For supported examples, default to only training on baseline abstentions.
     This avoids noisy pairs where baseline gives a semantically acceptable answer
     but EM marks it wrong.
  3. For multi-answer/list-like questions, output all supported gold answers,
     instead of always selecting the shortest answer.
  4. For alias-style gold answers, choose one concise supported answer.
  5. For unsupported examples, train the model to abstain when baseline answered
     without retrieved evidence.
"""

import argparse
import json
import os
import random
import re
import string
from collections import Counter


# -------------------------
# IO
# -------------------------


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -------------------------
# Normalization
# -------------------------


def normalize_for_eval(text):
    """
    NQ-style normalization for answer equivalence.

    Removes articles and punctuation.
    Use this for answer comparison, not support matching.
    """
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def normalize_for_support(text):
    """
    Stricter normalization for passage support matching.

    Important:
      Do NOT remove articles here.

    Example:
      "The Winans" should not become "winans",
      because that may falsely match "CeCe Winans".
    """
    if text is None:
        return ""

    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def token_span_match(normalized_text, normalized_answer):
    text_tokens = normalized_text.split()
    answer_tokens = normalized_answer.split()

    if not text_tokens or not answer_tokens:
        return False

    n = len(answer_tokens)
    if n > len(text_tokens):
        return False

    for i in range(len(text_tokens) - n + 1):
        if text_tokens[i : i + n] == answer_tokens:
            return True

    return False


def contains_answer_for_eval(text, answer):
    return token_span_match(
        normalize_for_eval(text),
        normalize_for_eval(answer),
    )


def contains_answer_for_support(text, answer):
    return token_span_match(
        normalize_for_support(text),
        normalize_for_support(answer),
    )


# -------------------------
# Answer utilities
# -------------------------


def get_all_gold_answers(example):
    answers = (
        example.get("answers")
        or example.get("gold_answers")
        or example.get("answer")
        or example.get("gold_answer")
        or []
    )

    if isinstance(answers, str):
        answer = answers.strip()
        return [answer] if answer else []

    if isinstance(answers, list):
        clean = []
        seen = set()

        for answer in answers:
            answer = str(answer).strip()
            if not answer:
                continue

            key = normalize_for_eval(answer)
            if key in seen:
                continue

            clean.append(answer)
            seen.add(key)

        return clean

    return []


def is_numeric_answer(answer):
    x = normalize_for_eval(answer)

    if not x:
        return False

    # Pure number, year, count, etc.
    if re.fullmatch(r"\d+", x):
        return True

    # Simple numeric phrase like "13 episodes" is also risky for weak support.
    tokens = x.split()
    if tokens and any(tok.isdigit() for tok in tokens):
        return True

    return False


def canonical_answer_for_equivalence(text):
    """
    Conservative alias heuristic to reduce noisy DPO pairs.

    This does NOT try to be a full semantic matcher.
    It only handles common short-answer variants.
    """
    x = normalize_for_eval(text)

    alias_map = {
        "runner up": "second",
        "runnerup": "second",
        "second place": "second",
        "2nd": "second",
        "came second": "second",
        "finished second": "second",
        "second": "second",
    }

    if x in alias_map:
        return alias_map[x]

    # Remove common title/honorific prefixes.
    title_prefixes = [
        "saint",
        "st",
        "sir",
        "dame",
        "lord",
        "lady",
        "king",
        "queen",
        "pope",
        "president",
        "prime minister",
        "dr",
        "doctor",
    ]

    for prefix in title_prefixes:
        if x.startswith(prefix + " "):
            x = x[len(prefix) + 1 :]
            break

    return x


def answers_look_like_aliases(gold_answers):
    """
    Decide whether multiple gold answers are likely aliases of one answer.

    Examples:
      ["14 December 1972 UTC", "December 1972"] -> aliases
      ["France", "Russia", "China"] -> not aliases
    """
    if len(gold_answers) <= 1:
        return True

    normalized = [normalize_for_eval(a) for a in gold_answers if normalize_for_eval(a)]
    if len(normalized) <= 1:
        return True

    # If all answers share substantial token overlap, treat as aliases.
    token_sets = [set(x.split()) for x in normalized]
    common = set.intersection(*token_sets) if token_sets else set()

    if common:
        return True

    # If one normalized answer is contained as a token span in another,
    # this is often an alias/shorter variant.
    for i, a in enumerate(normalized):
        for j, b in enumerate(normalized):
            if i == j:
                continue
            if token_span_match(b, a) or token_span_match(a, b):
                return True

    return False


def is_list_like_question(question, gold_answers):
    """
    Detect questions where multiple gold answers likely need to be output together.
    """
    if len(gold_answers) <= 1:
        return False

    q = normalize_for_eval(question)

    list_cues = [
        "what are",
        "what were",
        "who are",
        "who were",
        "which are",
        "which were",
        "name the",
        "list",
        "give me",
        "what five",
        "what 5",
        "which five",
        "which 5",
        "who five",
        "who 5",
    ]

    if any(cue in q for cue in list_cues):
        return True

    number_words = [
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    ]

    if re.search(r"\b\d+\b", q):
        return True

    if any(re.search(rf"\b{word}\b", q) for word in number_words):
        return True

    # Many distinct answers usually means a multi-answer case.
    if len(gold_answers) >= 4 and not answers_look_like_aliases(gold_answers):
        return True

    return False


def classify_answer_set(question, gold_answers):
    """
    Return:
      "single"       - one gold answer
      "alias"        - multiple surface forms of same answer
      "multi"        - multiple different answers may be needed
    """
    if len(gold_answers) <= 1:
        return "single"

    if is_list_like_question(question, gold_answers):
        return "multi"

    if answers_look_like_aliases(gold_answers):
        return "alias"

    # Multiple distinct answers, even if question is singular.
    # Example: multiple singers/actors may be acceptable.
    return "multi"


def join_answers(answers):
    answers = [str(a).strip() for a in answers if str(a).strip()]

    if not answers:
        return ""

    if len(answers) == 1:
        return answers[0]

    if len(answers) == 2:
        return f"{answers[0]} and {answers[1]}"

    return ", ".join(answers[:-1]) + f", and {answers[-1]}"


def baseline_contains_gold_count(text, gold_answers):
    count = 0

    for answer in gold_answers:
        if contains_answer_for_eval(text, answer):
            count += 1

    return count


def baseline_is_correct_or_equivalent(example, gold_answers, answer_set_type):
    """
    Robust correctness check for deciding whether baseline should be rejected.

    Uses:
      - EM from 05_eval_generation.py
      - alias heuristic
      - token containment in pred_answer
      - multi-answer coverage when appropriate
    """
    if float(example.get("em", 0.0)) > 0.0:
        return True

    pred_answer = str(example.get("pred_answer", "")).strip()
    generated_answer = str(example.get("generated_answer", "")).strip()

    pred_norm = canonical_answer_for_equivalence(pred_answer)

    for gold in gold_answers:
        gold_norm = canonical_answer_for_equivalence(gold)

        if pred_norm and pred_norm == gold_norm:
            return True

        # Handles cases like:
        #   pred = "Saint Matthias"
        #   gold = "Matthias"
        if pred_answer and contains_answer_for_eval(pred_answer, gold):
            return True

    if answer_set_type == "multi":
        pred_hits = baseline_contains_gold_count(pred_answer, gold_answers)
        gen_hits = baseline_contains_gold_count(generated_answer, gold_answers)

        # For explicit list-like questions, full coverage means correct.
        if is_list_like_question(str(example.get("question", "")), gold_answers):
            if pred_hits == len(gold_answers) or gen_hits == len(gold_answers):
                return True

        # For non-list multi-answer cases, if the baseline gave one accepted gold answer,
        # do not treat it as a bad rejected answer.
        if pred_hits >= 1:
            return True

    return False


# -------------------------
# Passage/prompt utilities
# -------------------------


def get_passage_id(passage, fallback_id):
    return passage.get("rank", passage.get("pid", passage.get("id", fallback_id)))


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

    for i, passage in enumerate(passages, start=1):
        pid = get_passage_id(passage, i)
        lines.append(f"[{pid}] {get_passage_text(passage)}")

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


# -------------------------
# Support selection
# -------------------------


def build_answer_support_map(passages, gold_answers):
    """
    Map each gold answer to passage ids that contain it under strict support matching.
    """
    answer_to_ids = {}

    for answer in gold_answers:
        ids = []

        for i, passage in enumerate(passages, start=1):
            pid = get_passage_id(passage, i)
            passage_text = get_passage_text(passage)

            if contains_answer_for_support(passage_text, answer):
                ids.append(pid)

        if ids:
            answer_to_ids[answer] = ids

    return answer_to_ids


def select_chosen_answer_and_support(
    question,
    passages,
    gold_answers,
    allow_numeric_supported,
):
    """
    Select chosen answer text and supporting passage ids.

    For alias/single answer:
      choose one supported answer.

    For multi-answer cases:
      output all supported gold answers, not just the shortest one.
    """
    answer_set_type = classify_answer_set(question, gold_answers)
    answer_to_ids = build_answer_support_map(passages, gold_answers)

    if not answer_to_ids:
        return "", [], answer_set_type, []

    if answer_set_type == "multi":
        supported_answers = [a for a in gold_answers if a in answer_to_ids]

        # For explicit list-like questions, require every listed gold answer to be supported.
        # Otherwise we risk training incomplete list answers.
        if is_list_like_question(question, gold_answers):
            if len(supported_answers) < len(gold_answers):
                return "", [], answer_set_type, supported_answers

        if not allow_numeric_supported:
            if any(is_numeric_answer(a) for a in supported_answers):
                return "", [], answer_set_type, supported_answers

        chosen_answer = join_answers(supported_answers)

        # Prefer one passage that supports all selected answers if available.
        all_ids = []
        for ids in answer_to_ids.values():
            all_ids.extend(ids)

        unique_ids = []
        for pid in all_ids:
            if pid not in unique_ids:
                unique_ids.append(pid)

        return chosen_answer, unique_ids, answer_set_type, supported_answers

    # single / alias case
    supported_answers = [a for a in gold_answers if a in answer_to_ids]

    # Choose a concise supported answer for alias-style answers.
    supported_answers.sort(key=lambda x: (len(x.split()), len(x)))
    chosen_answer = supported_answers[0]

    if not allow_numeric_supported and is_numeric_answer(chosen_answer):
        return "", [], answer_set_type, supported_answers

    return chosen_answer, answer_to_ids[chosen_answer], answer_set_type, supported_answers


def build_supported_chosen(chosen_answer, supporting_passage_ids):
    citation_text = " ".join(f"[{pid}]" for pid in supporting_passage_ids[:3])

    return (
        f"Answer: {chosen_answer}\n"
        f"Evidence: The answer is supported by passage {citation_text}."
    )


def build_unsupported_chosen():
    return (
        "Answer: Insufficient evidence\n"
        "Evidence: The provided passages do not contain enough information to infer the answer."
    )


# -------------------------
# Labels
# -------------------------


def is_abstention(example):
    if "abstained" in example:
        return bool(example["abstained"])

    if "is_abstention" in example:
        return bool(example["is_abstention"])

    output = normalize_for_eval(example.get("generated_answer", ""))

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

    return any(pattern in output for pattern in abstention_patterns)


def is_supported(example):
    if "answer_in_retrieved" in example:
        return bool(example["answer_in_retrieved"])

    for key in ["is_supported", "retrieved_answer_supported", "supported"]:
        if key in example:
            return bool(example[key])

    return False


# -------------------------
# Sampling
# -------------------------


def select_final_pairs(
    supported_pairs,
    unsupported_pairs,
    supported_ratio,
    max_unsupported_per_supported,
):
    """
    Select final DPO rows.

    We keep all clean supported pairs when possible, then downsample unsupported
    pairs so DPO does not become too abstention-heavy.
    """
    random.shuffle(supported_pairs)
    random.shuffle(unsupported_pairs)

    if not supported_pairs and not unsupported_pairs:
        return []

    if not supported_pairs:
        return unsupported_pairs

    if supported_ratio <= 0:
        selected = unsupported_pairs
        random.shuffle(selected)
        return selected

    # Use all supported pairs.
    selected_supported = supported_pairs

    # Target unsupported count implied by supported_ratio:
    # supported / (supported + unsupported) = supported_ratio
    target_unsupported = int(len(selected_supported) * (1.0 - supported_ratio) / supported_ratio)

    if max_unsupported_per_supported is not None and max_unsupported_per_supported >= 0:
        capped_unsupported = int(len(selected_supported) * max_unsupported_per_supported)
        target_unsupported = min(target_unsupported, capped_unsupported)

    target_unsupported = max(0, target_unsupported)
    selected_unsupported = unsupported_pairs[: min(len(unsupported_pairs), target_unsupported)]

    rows = selected_supported + selected_unsupported
    random.shuffle(rows)
    return rows


# -------------------------
# Main construction
# -------------------------


def build_dpo_data(
    eval_file,
    output_file,
    stats_file,
    supported_ratio,
    supported_mode,
    allow_numeric_supported,
    max_unsupported_per_supported,
    seed,
):
    random.seed(seed)

    supported_pairs = []
    unsupported_pairs = []
    stats = Counter()

    for ex in read_jsonl(eval_file):
        stats["num_examples"] += 1

        question = str(ex.get("question", "")).strip()
        passages = ex.get("retrieved_passages") or ex.get("passages") or []
        rejected = str(ex.get("generated_answer", "")).strip()
        gold_answers = get_all_gold_answers(ex)

        if not question:
            stats["skipped_missing_question"] += 1
            continue

        if not passages:
            stats["skipped_missing_passages"] += 1
            continue

        if not rejected:
            stats["skipped_missing_rejected"] += 1
            continue

        if not gold_answers:
            stats["skipped_missing_gold_answers"] += 1
            continue

        prompt = ex.get("prompt") or build_prompt(question, passages)

        supported = is_supported(ex)
        abstained = is_abstention(ex)

        answer_set_type = classify_answer_set(question, gold_answers)
        correct = baseline_is_correct_or_equivalent(ex, gold_answers, answer_set_type)
        em_correct = float(ex.get("em", 0.0)) > 0.0

        if correct and not em_correct:
            stats["non_em_but_equivalent_or_contained"] += 1

        if supported:
            stats["supported_examples"] += 1

            chosen_answer, supporting_ids, answer_set_type, supported_gold_answers = (
                select_chosen_answer_and_support(
                    question=question,
                    passages=passages,
                    gold_answers=gold_answers,
                    allow_numeric_supported=allow_numeric_supported,
                )
            )

            if not chosen_answer or not supporting_ids:
                stats["supported_missing_or_filtered_support"] += 1

                if any(is_numeric_answer(a) for a in gold_answers):
                    stats["supported_numeric_filtered"] += 1

                continue

            if answer_set_type == "multi":
                stats["supported_multi_answer_candidates"] += 1

            # Default high-quality mode:
            # only use supported examples where baseline abstained.
            if supported_mode == "abstention_only":
                if not abstained:
                    stats["supported_non_abstain_skipped"] += 1
                    continue
            else:
                # Less conservative mode:
                # use abstained or clearly incorrect examples.
                if correct:
                    stats["supported_already_correct"] += 1
                    continue

            chosen = build_supported_chosen(chosen_answer, supporting_ids)

            supported_pairs.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "type": "supported_answer",
                    "id": ex.get("id"),
                    "question": question,
                    "chosen_answer": chosen_answer,
                    "gold_answers": gold_answers,
                    "supported_gold_answers": supported_gold_answers,
                    "answer_set_type": answer_set_type,
                    "supporting_passage_ids": supporting_ids,
                    "baseline_pred_answer": ex.get("pred_answer"),
                    "baseline_abstained": abstained,
                    "baseline_correct": correct,
                    "baseline_em_correct": em_correct,
                    "answer_in_retrieved": supported,
                    "baseline_citation_ids": ex.get("citation_ids", []),
                }
            )

            stats["supported_pairs"] += 1

        else:
            stats["unsupported_examples"] += 1

            if correct:
                stats["unsupported_but_correct_skipped"] += 1
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
                    "id": ex.get("id"),
                    "question": question,
                    "gold_answers": gold_answers,
                    "answer_set_type": answer_set_type,
                    "baseline_pred_answer": ex.get("pred_answer"),
                    "baseline_abstained": abstained,
                    "baseline_correct": correct,
                    "baseline_em_correct": em_correct,
                    "answer_in_retrieved": supported,
                    "baseline_citation_ids": ex.get("citation_ids", []),
                }
            )

            stats["unsupported_pairs"] += 1

    dpo_rows = select_final_pairs(
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
        row["type"] == "supported_answer" for row in dpo_rows
    )
    stats["final_unsupported_pairs"] = sum(
        row["type"] == "unsupported_abstention" for row in dpo_rows
    )

    if stats_file:
        stats_dir = os.path.dirname(stats_file)
        if stats_dir:
            os.makedirs(stats_dir, exist_ok=True)

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
        default=0.5,
        help=(
            "Target fraction of supported pairs in final DPO data. "
            "Recommended: 0.5 for balanced supported/unsupported training."
        ),
    )

    parser.add_argument(
        "--supported_mode",
        choices=["abstention_only", "wrong_and_abstention"],
        default="abstention_only",
        help=(
            "abstention_only is cleaner and recommended. "
            "wrong_and_abstention includes supported wrong non-abstain examples, "
            "but may introduce more noise."
        ),
    )

    parser.add_argument(
        "--allow_numeric_supported",
        action="store_true",
        help=(
            "Allow supported pairs with numeric answers. "
            "Default is to skip them because numeric answers often create false "
            "positive support under weak string matching."
        ),
    )

    parser.add_argument(
        "--max_unsupported_per_supported",
        type=float,
        default=1.0,
        help=(
            "Cap final unsupported pairs relative to supported pairs. "
            "Use 1.0 for roughly balanced data."
        ),
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
        supported_mode=args.supported_mode,
        allow_numeric_supported=args.allow_numeric_supported,
        max_unsupported_per_supported=args.max_unsupported_per_supported,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
