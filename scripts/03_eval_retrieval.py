from pathlib import Path
import json
import re


RETRIEVAL_PATH = Path("data/retrieval/nq_validation_bm25_top5.jsonl")


def normalize(text: str) -> str:
    """Normalize text for simple string matching."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def answer_in_retrieved_passages(answers: list[str], passages: list[dict]) -> bool:
    """Check whether any gold answer appears in the retrieved passages."""
    context = " ".join(
        f"{passage['title']} {passage['text']}"
        for passage in passages
    )
    context = normalize(context)

    for answer in answers:
        if normalize(answer) in context:
            return True

    return False


def main() -> None:
    total = 0
    hit = 0

    with RETRIEVAL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            total += 1

            if answer_in_retrieved_passages(
                example["answers"],
                example["retrieved_passages"]
            ):
                hit += 1

    recall = hit / total if total > 0 else 0.0

    print(f"Total examples: {total}")
    print(f"Recall@5: {recall:.4f} ({hit}/{total})")


if __name__ == "__main__":
    main()