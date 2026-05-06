#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM verifier audit for RAG hallucination evaluation.

This script uses an external LLM judge to evaluate whether model outputs are:
1. correct with respect to gold answers,
2. supported by retrieved passages,
3. citation-faithful,
4. hallucinated.

It evaluates Baseline / DPO-v1 / DPO-v2 side by side for the same examples.

This is an audit script, not a training script.

Example with ChatECNU:

  export OPENAI_API_KEY="your_ecnu_api_key"
  export OPENAI_BASE_URL="https://chat.ecnu.edu.cn/open/api/v1"

  python scripts/09_llm_verifier_audit.py \
    --baseline_eval outputs/baseline/base_val_eval_top10_full.jsonl \
    --dpo_v1_eval outputs/dpo/dpo_val_eval_top10_full.jsonl \
    --dpo_v2_eval outputs/dpo/dpo_v2_u2_lightclean_val_eval_top10_full.jsonl \
    --output_file results/llm_verifier_audit_unsupported_sample300_ecnu_v2.jsonl \
    --summary_file results/llm_verifier_audit_unsupported_sample300_ecnu_v2_summary.json \
    --sample_size 300 \
    --split unsupported \
    --model ecnu-plus \
    --base_url https://chat.ecnu.edu.cn/open/api/v1 \
    --resume
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


SYSTEM_PROMPT = """You are a strict evaluator for retrieval-augmented question answering.

You will be given:
- a question,
- gold answers,
- retrieved passages,
- outputs from three systems.

Your job is to judge each system output.

Important rules:
1. Judge passage support ONLY from the provided retrieved passages.
2. Do not use outside knowledge to decide passage support.
3. Gold answers are used only as reference for answer correctness.
4. A correct answer is not necessarily passage-supported.
5. A passage-supported answer is not necessarily equal to the provided gold answer if the dataset label is noisy.
6. Citation faithfulness means the cited passages actually support the answer, not merely that the citation ID exists.
7. If the model says "Insufficient evidence", treat it as an abstention.
8. If the model abstains but still cites one or more passages, citation_faithfulness must be "unfaithful", because an insufficient-evidence answer should not cite supporting passages.
9. Use "no_citation" only when the system output contains no citation markers like [1], [2], or [10].
10. If an answer is unsupported by the provided passages and the model nevertheless gives a factual answer, hallucination should be "yes" or "partial".
11. If the model correctly abstains because passages do not contain enough evidence, hallucination should be "no".
12. If the model gives a supported answer with faithful citations, hallucination should be "no".
13. Be strict about citation faithfulness: cited passages must support the answer, not merely be topically related.

Return valid JSON only. Do not include markdown.
"""


VALID_LABELS = {
    "answer_type": {"answer", "abstention", "malformed"},
    "gold_correctness": {"correct", "partially_correct", "incorrect", "cannot_judge"},
    "passage_support": {"supported", "partially_supported", "unsupported", "not_applicable"},
    "citation_faithfulness": {
        "faithful",
        "partially_faithful",
        "unfaithful",
        "no_citation",
        "not_applicable",
    },
    "hallucination": {"yes", "no", "partial", "not_applicable"},
}


def read_jsonl_as_dict(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            rows[str(row["id"])] = row

    return rows


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_text(x: Any) -> str:
    return str(x or "").strip()


def compact_passages(
    passages: List[Dict[str, Any]],
    max_passages: int,
    max_chars_per_passage: int,
) -> List[Dict[str, Any]]:
    compact = []

    for i, p in enumerate(passages[:max_passages], start=1):
        rank = p.get("rank", i)
        title = normalize_text(p.get("title"))
        text = normalize_text(p.get("text") or p.get("contents") or p.get("passage"))

        if len(text) > max_chars_per_passage:
            text = text[:max_chars_per_passage] + " ..."

        compact.append(
            {
                "rank": rank,
                "title": title,
                "text": text,
            }
        )

    return compact


def has_citation_markers(text: Any) -> bool:
    return bool(re.search(r"\[\d+\]", str(text or "")))


def parse_citation_markers(text: Any) -> List[int]:
    return sorted(set(int(x) for x in re.findall(r"\[(\d+)\]", str(text or ""))))


def normalize_abstention_text(text: Any) -> str:
    x = str(text or "").lower()
    x = re.sub(r"[^a-z0-9 ]+", " ", x)
    x = " ".join(x.split())
    return x


def looks_like_abstention(text: Any) -> bool:
    x = normalize_abstention_text(text)

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


def build_user_prompt(
    question: str,
    gold_answers: List[str],
    passages: List[Dict[str, Any]],
    outputs: Dict[str, Dict[str, Any]],
) -> str:
    payload = {
        "question": question,
        "gold_answers": gold_answers,
        "retrieved_passages": passages,
        "systems": outputs,
        "evaluation_schema": {
            "answer_type": (
                "answer | abstention | malformed. "
                "Use abstention if the system says Insufficient evidence or cannot answer."
            ),
            "gold_correctness": (
                "correct | partially_correct | incorrect | cannot_judge. "
                "Judge whether the answer matches the gold answers. "
                "If the system abstains, use cannot_judge."
            ),
            "passage_support": (
                "supported | partially_supported | unsupported | not_applicable. "
                "Judge whether the answer is supported by the retrieved passages. "
                "If the system abstains, use not_applicable."
            ),
            "citation_faithfulness": (
                "faithful | partially_faithful | unfaithful | no_citation | not_applicable. "
                "If the output contains no citation markers, use no_citation. "
                "If the output contains citation markers but the answer is an abstention, use unfaithful. "
                "If the cited passages do not support the answer, use unfaithful."
            ),
            "hallucination": (
                "yes | no | partial | not_applicable. "
                "Use yes if the model gives an unsupported factual answer. "
                "Use no if the model correctly abstains or gives a supported answer."
            ),
            "brief_reason": "short explanation",
        },
        "output_format": {
            "baseline": {
                "answer_type": "...",
                "gold_correctness": "...",
                "passage_support": "...",
                "citation_faithfulness": "...",
                "hallucination": "...",
                "brief_reason": "...",
            },
            "dpo_v1": {
                "answer_type": "...",
                "gold_correctness": "...",
                "passage_support": "...",
                "citation_faithfulness": "...",
                "hallucination": "...",
                "brief_reason": "...",
            },
            "dpo_v2": {
                "answer_type": "...",
                "gold_correctness": "...",
                "passage_support": "...",
                "citation_faithfulness": "...",
                "hallucination": "...",
                "brief_reason": "...",
            },
        },
    }

    return json.dumps(payload, ensure_ascii=False, indent=2)


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"No JSON object found in model response: {text[:500]}")

    return json.loads(match.group(0))


def normalize_judge_labels(judge: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make judge output robust against minor label variants.
    """
    systems = ["baseline", "dpo_v1", "dpo_v2"]

    label_aliases = {
        "answer_type": {
            "abstain": "abstention",
            "abstained": "abstention",
            "insufficient_evidence": "abstention",
            "insufficient evidence": "abstention",
            "answers": "answer",
        },
        "gold_correctness": {
            "partially correct": "partially_correct",
            "cannot judge": "cannot_judge",
            "not_applicable": "cannot_judge",
            "n/a": "cannot_judge",
        },
        "passage_support": {
            "partially supported": "partially_supported",
            "not applicable": "not_applicable",
            "n/a": "not_applicable",
        },
        "citation_faithfulness": {
            "partially faithful": "partially_faithful",
            "no citation": "no_citation",
            "no citations": "no_citation",
            "not applicable": "not_applicable",
            "n/a": "not_applicable",
            "invalid": "unfaithful",
            "inappropriate": "unfaithful",
            "inappropriate_citation": "unfaithful",
        },
        "hallucination": {
            "false": "no",
            "true": "yes",
            "none": "no",
            "not applicable": "not_applicable",
            "n/a": "not_applicable",
        },
    }

    for system in systems:
        if system not in judge or not isinstance(judge[system], dict):
            judge[system] = {}

        for key, valid in VALID_LABELS.items():
            value = judge[system].get(key, "missing")

            if value is None:
                value = "missing"

            value = str(value).strip().lower()
            value = label_aliases.get(key, {}).get(value, value)

            if value not in valid:
                value = "missing"

            judge[system][key] = value

        if "brief_reason" not in judge[system]:
            judge[system]["brief_reason"] = ""

    return judge


def apply_deterministic_citation_overrides(
    judge: Dict[str, Any],
    outputs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Fix obvious citation-labeling mistakes from the LLM judge.

    Key rule:
    If a system abstains but still emits citation markers, this is not no_citation.
    It is inappropriate / unfaithful citation behavior.
    """
    for system_name, system_output in outputs.items():
        generated = system_output.get("generated_answer", "")
        pred = system_output.get("pred_answer", "")
        cites = parse_citation_markers(generated)
        has_cites = len(cites) > 0

        if system_name not in judge:
            continue

        answer_type = judge[system_name].get("answer_type")

        # If the automatic output looks like abstention, force answer_type to abstention.
        if looks_like_abstention(generated) or looks_like_abstention(pred):
            if answer_type != "abstention":
                judge[system_name]["answer_type_original"] = answer_type
                judge[system_name]["answer_type"] = "abstention"
                judge[system_name]["answer_type_override_applied"] = True

        answer_type = judge[system_name].get("answer_type")

        # The key fix:
        # "Insufficient evidence" with [1] [2] is an unfaithful / inappropriate citation.
        if has_cites and answer_type == "abstention":
            original = judge[system_name].get("citation_faithfulness")
            judge[system_name]["citation_faithfulness_original"] = original
            judge[system_name]["citation_faithfulness"] = "unfaithful"
            judge[system_name]["citation_override_applied"] = True
            judge[system_name]["citation_markers"] = cites
            judge[system_name]["brief_reason"] = (
                str(judge[system_name].get("brief_reason", "")).strip()
                + " Citation override: the system abstained but still emitted citation markers."
            ).strip()

        # If citation markers exist, no_citation is impossible.
        elif has_cites and judge[system_name].get("citation_faithfulness") == "no_citation":
            original = judge[system_name].get("citation_faithfulness")
            judge[system_name]["citation_faithfulness_original"] = original
            judge[system_name]["citation_fathfulness_note"] = (
                "Judge said no_citation despite citation markers."
            )
            judge[system_name]["citation_faithfulness"] = "unfaithful"
            judge[system_name]["citation_override_applied"] = True
            judge[system_name]["citation_markers"] = cites

        # If no citation markers exist, force no_citation unless judge used not_applicable.
        elif not has_cites:
            cf = judge[system_name].get("citation_faithfulness")
            if cf in {"faithful", "partially_faithful", "unfaithful"}:
                judge[system_name]["citation_faithfulness_original"] = cf
                judge[system_name]["citation_faithfulness"] = "no_citation"
                judge[system_name]["citation_override_applied"] = True
                judge[system_name]["citation_markers"] = []

    return judge


def call_judge(
    client: OpenAI,
    model: str,
    user_prompt: str,
    temperature: float,
    max_retries: int,
    sleep_seconds: float,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
            )
            content = response.choices[0].message.content or ""
            judge = extract_json_object(content)
            judge = normalize_judge_labels(judge)
            return judge

        except Exception as e:
            last_error = e
            wait = sleep_seconds * (2 ** attempt)
            print(f"[WARN] Judge call failed on attempt {attempt + 1}/{max_retries}: {e}")
            print(f"[WARN] Sleeping {wait:.1f}s before retry.")
            time.sleep(wait)

    raise RuntimeError(f"Judge call failed after {max_retries} retries: {last_error}")


def choose_ids(
    baseline: Dict[str, Dict[str, Any]],
    dpo_v1: Dict[str, Dict[str, Any]],
    dpo_v2: Dict[str, Dict[str, Any]],
    split: str,
    sample_size: int,
    seed: int,
) -> List[str]:
    ids = sorted(set(baseline) & set(dpo_v1) & set(dpo_v2))

    if split == "supported":
        ids = [i for i in ids if bool(baseline[i].get("answer_in_retrieved"))]
    elif split == "unsupported":
        ids = [i for i in ids if not bool(baseline[i].get("answer_in_retrieved"))]
    elif split == "all":
        pass
    else:
        raise ValueError("--split must be one of: all, supported, unsupported")

    random.seed(seed)
    random.shuffle(ids)

    if sample_size > 0:
        ids = ids[:sample_size]

    return ids


def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    systems = ["baseline", "dpo_v1", "dpo_v2"]

    summary: Dict[str, Any] = {
        "num_examples": len(rows),
        "systems": {},
    }

    for system in systems:
        counters = defaultdict(Counter)

        for row in rows:
            judge = row.get("judge", {}).get(system, {})

            for key in [
                "answer_type",
                "gold_correctness",
                "passage_support",
                "citation_faithfulness",
                "hallucination",
            ]:
                counters[key][judge.get(key, "missing")] += 1

            if judge.get("citation_override_applied"):
                counters["overrides"]["citation_override_applied"] += 1

            if judge.get("answer_type_override_applied"):
                counters["overrides"]["answer_type_override_applied"] += 1

        system_summary = {}

        for key, counter in counters.items():
            total = sum(counter.values())
            system_summary[key] = {
                label: {
                    "count": count,
                    "rate": count / total if total else 0.0,
                }
                for label, count in sorted(counter.items())
            }

        total = len(rows)

        if total:
            answer_count = counters["answer_type"]["answer"]
            abstention_count = counters["answer_type"]["abstention"]

            supported_count = counters["passage_support"]["supported"]
            partially_supported_count = counters["passage_support"]["partially_supported"]
            unsupported_count = counters["passage_support"]["unsupported"]

            faithful_count = counters["citation_faithfulness"]["faithful"]
            partially_faithful_count = counters["citation_faithfulness"]["partially_faithful"]
            unfaithful_count = counters["citation_faithfulness"]["unfaithful"]
            no_citation_count = counters["citation_faithfulness"]["no_citation"]

            hallucination_yes = counters["hallucination"]["yes"]
            hallucination_partial = counters["hallucination"]["partial"]
            hallucination_no = counters["hallucination"]["no"]

            system_summary["derived_rates"] = {
                "answer_rate": answer_count / total,
                "abstention_rate": abstention_count / total,
                "supported_rate": supported_count / total,
                "partial_or_supported_rate": (
                    supported_count + partially_supported_count
                ) / total,
                "unsupported_rate": unsupported_count / total,
                "faithful_citation_rate": faithful_count / total,
                "partial_or_faithful_citation_rate": (
                    faithful_count + partially_faithful_count
                ) / total,
                "unfaithful_citation_rate": unfaithful_count / total,
                "no_citation_rate": no_citation_count / total,
                "hallucination_rate": hallucination_yes / total,
                "partial_or_hallucination_rate": (
                    hallucination_yes + hallucination_partial
                ) / total,
                "non_hallucination_rate": hallucination_no / total,
            }

        summary["systems"][system] = system_summary

    return summary


def build_system_outputs(
    baseline_row: Dict[str, Any],
    dpo_v1_row: Dict[str, Any],
    dpo_v2_row: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return {
        "baseline": {
            "generated_answer": baseline_row.get("generated_answer", ""),
            "pred_answer": baseline_row.get("pred_answer", ""),
            "automatic_em": baseline_row.get("em"),
            "automatic_f1": baseline_row.get("f1"),
            "automatic_abstained": baseline_row.get("abstained"),
            "automatic_citation_ids": baseline_row.get("citation_ids"),
        },
        "dpo_v1": {
            "generated_answer": dpo_v1_row.get("generated_answer", ""),
            "pred_answer": dpo_v1_row.get("pred_answer", ""),
            "automatic_em": dpo_v1_row.get("em"),
            "automatic_f1": dpo_v1_row.get("f1"),
            "automatic_abstained": dpo_v1_row.get("abstained"),
            "automatic_citation_ids": dpo_v1_row.get("citation_ids"),
        },
        "dpo_v2": {
            "generated_answer": dpo_v2_row.get("generated_answer", ""),
            "pred_answer": dpo_v2_row.get("pred_answer", ""),
            "automatic_em": dpo_v2_row.get("em"),
            "automatic_f1": dpo_v2_row.get("f1"),
            "automatic_abstained": dpo_v2_row.get("abstained"),
            "automatic_citation_ids": dpo_v2_row.get("citation_ids"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--baseline_eval", type=Path, required=True)
    parser.add_argument("--dpo_v1_eval", type=Path, required=True)
    parser.add_argument("--dpo_v2_eval", type=Path, required=True)

    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--summary_file", type=Path, required=True)

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base_url", type=str, default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY"))

    parser.add_argument(
        "--sample_size",
        type=int,
        default=300,
        help="Number of examples to audit. Use <=0 for all examples in the selected split.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="unsupported",
        choices=["all", "supported", "unsupported"],
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_passages", type=int, default=10)
    parser.add_argument("--max_chars_per_passage", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--sleep_seconds", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError(
            "Missing API key. Set OPENAI_API_KEY or pass --api_key. "
            "Do not commit API keys to GitHub."
        )

    client_kwargs = {"api_key": args.api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url

    client = OpenAI(**client_kwargs)

    baseline = read_jsonl_as_dict(args.baseline_eval)
    dpo_v1 = read_jsonl_as_dict(args.dpo_v1_eval)
    dpo_v2 = read_jsonl_as_dict(args.dpo_v2_eval)

    ids = choose_ids(
        baseline=baseline,
        dpo_v1=dpo_v1,
        dpo_v2=dpo_v2,
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    done_ids = set()
    rows: List[Dict[str, Any]] = []

    if args.resume and args.output_file.exists():
        with args.output_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows.append(row)
                    done_ids.add(str(row["id"]))

        print(f"[resume] Loaded {len(done_ids)} existing judged examples.")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.resume else "w"

    with args.output_file.open(mode, encoding="utf-8") as out_f:
        for idx, ex_id in enumerate(ids, start=1):
            if ex_id in done_ids:
                continue

            b = baseline[ex_id]
            v1 = dpo_v1[ex_id]
            v2 = dpo_v2[ex_id]

            question = b.get("question", "")
            gold_answers = b.get("answers", [])
            passages = compact_passages(
                b.get("retrieved_passages", []),
                max_passages=args.max_passages,
                max_chars_per_passage=args.max_chars_per_passage,
            )

            outputs = build_system_outputs(
                baseline_row=b,
                dpo_v1_row=v1,
                dpo_v2_row=v2,
            )

            user_prompt = build_user_prompt(
                question=question,
                gold_answers=gold_answers,
                passages=passages,
                outputs=outputs,
            )

            judge = call_judge(
                client=client,
                model=args.model,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_retries=args.max_retries,
                sleep_seconds=args.sleep_seconds,
            )

            judge = apply_deterministic_citation_overrides(judge, outputs)

            row = {
                "id": ex_id,
                "split": args.split,
                "question": question,
                "gold_answers": gold_answers,
                "answer_in_retrieved_weak": b.get("answer_in_retrieved"),
                "retrieved_passages": passages,
                "systems": outputs,
                "judge": judge,
            }

            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

            rows.append(row)

            print(f"[{idx}/{len(ids)}] judged id={ex_id}")

    summary = summarize_results(rows)
    write_json(args.summary_file, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
