from pathlib import Path
import json
import pickle
import re

import numpy as np
from tqdm import tqdm


NQ_PATH = Path("data/raw/nq_open/validation.jsonl")
INDEX_DIR = Path("indexes/bm25")
OUT_DIR = Path("data/retrieval")


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 retrieval."""
    return re.findall(r"\w+", text.lower())


def load_jsonl(path: Path):
    """Load a JSONL file line by line."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main() -> None:
    top_k = 5
    max_examples = 500

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with (INDEX_DIR / "bm25.pkl").open("rb") as f:
        bm25 = pickle.load(f)

    with (INDEX_DIR / "passages.pkl").open("rb") as f:
        passages = pickle.load(f)

    out_path = OUT_DIR / f"nq_validation_bm25_top{top_k}.jsonl"

    with out_path.open("w", encoding="utf-8") as fout:
        for idx, example in enumerate(tqdm(load_jsonl(NQ_PATH), desc="Retrieving")):
            if idx >= max_examples:
                break

            question = example["question"]
            answers = example["answer"]

            scores = bm25.get_scores(tokenize(question))
            top_indices = np.argsort(scores)[::-1][:top_k]

            retrieved_passages = []

            for rank, passage_idx in enumerate(top_indices, start=1):
                passage = passages[int(passage_idx)]

                retrieved_passages.append(
                    {
                        "rank": rank,
                        "score": float(scores[passage_idx]),
                        "pid": passage["pid"],
                        "old_id": passage.get("old_id"),
                        "title": passage["title"],
                        "text": passage["text"],
                    }
                )

            record = {
                "id": idx,
                "question": question,
                "answers": answers,
                "retrieved_passages": retrieved_passages,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved retrieval results to {out_path}")


if __name__ == "__main__":
    main()
