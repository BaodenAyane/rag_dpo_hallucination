from pathlib import Path
import json
import pickle
import re

from rank_bm25 import BM25Okapi
from tqdm import tqdm


RAW_WIKI_PATH = Path("data/raw/wiki_dpr/wiki_dpr_first_1_shards_text_only.jsonl")
INDEX_DIR = Path("indexes/bm25")


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def load_passages(path: Path):
    passages = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line))
    return passages


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading passages from {RAW_WIKI_PATH}")
    passages = load_passages(RAW_WIKI_PATH)

    print(f"Loaded {len(passages)} passages")

    corpus_tokens = []
    for p in tqdm(passages, desc="Tokenizing passages"):
        text = f"{p['title']} {p['text']}"
        corpus_tokens.append(tokenize(text))

    print("Building BM25 index...")
    bm25 = BM25Okapi(corpus_tokens)

    with (INDEX_DIR / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)

    with (INDEX_DIR / "passages.pkl").open("wb") as f:
        pickle.dump(passages, f)

    print(f"Saved BM25 index to {INDEX_DIR}")


if __name__ == "__main__":
    main()
