from pathlib import Path
import json
import pickle
import re

from rank_bm25 import BM25Okapi
from tqdm import tqdm


RAW_WIKI_PATH = Path("data/raw/wiki_dpr/wiki_dpr_first_1_shards_text_only.jsonl")
INDEX_DIR = Path("indexes/bm25")


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 using a simple lowercase word tokenizer."""
    return re.findall(r"\w+", text.lower())


def load_passages(path: Path) -> list[dict]:
    """Load Wikipedia passages from a JSONL file."""
    passages = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line))

    return passages


def build_corpus_tokens(passages: list[dict]) -> list[list[str]]:
    """Build tokenized BM25 corpus from passage title and text."""
    corpus_tokens = []

    for passage in tqdm(passages, desc="Tokenizing passages"):
        # Only title and text are indexed. Passage ids are kept as metadata.
        text = f"{passage.get('title', '')} {passage.get('text', '')}"
        corpus_tokens.append(tokenize(text))

    return corpus_tokens


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading passages from {RAW_WIKI_PATH}")
    passages = load_passages(RAW_WIKI_PATH)
    print(f"Loaded {len(passages)} passages")

    corpus_tokens = build_corpus_tokens(passages)

    print("Building BM25 index...")
    bm25 = BM25Okapi(corpus_tokens)

    # Save both the BM25 object and passage metadata for retrieval.
    with (INDEX_DIR / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)

    with (INDEX_DIR / "passages.pkl").open("wb") as f:
        pickle.dump(passages, f)

    print(f"Saved BM25 index to {INDEX_DIR}")


if __name__ == "__main__":
    main()
