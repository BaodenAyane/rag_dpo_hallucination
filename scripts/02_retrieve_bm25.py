"""
Retrieve top-k passages for NQ-Open questions using a Pyserini/Lucene BM25 index.

Before running this script, build the index with 01_build_bm25_index.py.

Example:
  python scripts/02_retrieve_bm25.py \
    --questions data/raw/nq_open/validation.jsonl \
    --index_dir indexes/bm25/wiki_dpr_50_shards \
    --output data/retrieval/nq_validation_bm25_top10_50_shards.jsonl \
    --top_k 10 \
    --max_questions 500

Optional BM25 parameters:
  python scripts/02_retrieve_bm25.py \
    --questions data/raw/nq_open/validation.jsonl \
    --index_dir indexes/bm25/wiki_dpr_50_shards \
    --output data/retrieval/nq_validation_bm25_top10_50_shards.jsonl \
    --top_k 10 \
    --max_questions 500 \
    --bm25_k1 0.9 \
    --bm25_b 0.4

Input question JSONL format:
{
  "id": "0",
  "question": "...",
  "answers": ["..."]
}

Output JSONL format:
{
  "id": "0",
  "question": "...",
  "answers": ["..."],
  "retrieved_passages": [
    {
      "rank": 1,
      "score": 12.34,
      "passage_id": "wiki_123",
      "source_id": "123",
      "title": "...",
      "text": "..."
    }
  ]
}
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream records from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_answers(record: Dict[str, Any]) -> List[str]:
    """Normalize answer fields to a list of strings."""
    answers = record.get("answers", record.get("answer", []))

    if isinstance(answers, str):
        return [answers]

    if isinstance(answers, list):
        return [str(a) for a in answers]

    return []


def parse_raw_doc(raw: str) -> Dict[str, Any]:
    """
    Parse raw document stored in the Pyserini index.

    In 01_build_bm25_index.py, we stored raw docs like:
    {
      "id": "wiki_0",
      "contents": "Aaron: Aaron Aaron ...",
      "source_id": "1",
      "title": "Aaron",
      "text": "Aaron Aaron ..."
    }
    """
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "id": None,
            "source_id": None,
            "title": "",
            "text": raw,
        }

    title = str(doc.get("title", "")).strip()
    text = str(doc.get("text", "")).strip()

    # Fallback for indexes that only stored {"id": ..., "contents": ...}.
    if not text:
        contents = str(doc.get("contents", "")).strip()
        if ": " in contents:
            maybe_title, maybe_text = contents.split(": ", 1)
            if not title:
                title = maybe_title.strip()
            text = maybe_text.strip()
        else:
            text = contents

    return {
        "id": doc.get("id"),
        "source_id": doc.get("source_id"),
        "title": title,
        "text": text,
    }


def retrieve_topk(
    searcher: LuceneSearcher,
    question: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Retrieve top-k passages for a single question using Pyserini BM25."""
    hits = searcher.search(question, k=top_k)

    retrieved_passages = []

    for rank, hit in enumerate(hits, start=1):
        lucene_doc = searcher.doc(hit.docid)

        if lucene_doc is None:
            continue

        raw = lucene_doc.raw()
        parsed = parse_raw_doc(raw)

        retrieved_passages.append(
            {
                "rank": rank,
                "score": float(hit.score),
                "passage_id": parsed.get("id") or hit.docid,
                "source_id": parsed.get("source_id"),
                "title": parsed.get("title", ""),
                "text": parsed.get("text", ""),
            }
        )

    return retrieved_passages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--index_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_questions", type=int, default=None)

    # Pyserini/Anserini commonly uses these defaults for BM25 in many examples.
    # You can tune them later, but keep fixed for reproducibility.
    parser.add_argument("--bm25_k1", type=float, default=0.9)
    parser.add_argument("--bm25_b", type=float, default=0.4)

    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading Pyserini BM25 index from: {args.index_dir}")
    searcher = LuceneSearcher(str(args.index_dir))
    searcher.set_bm25(k1=args.bm25_k1, b=args.bm25_b)
    print(f"BM25 parameters: k1={args.bm25_k1}, b={args.bm25_b}")

    with args.output.open("w", encoding="utf-8") as fout:
        for idx, example in enumerate(tqdm(read_jsonl(args.questions), desc="Retrieving")):
            if args.max_questions is not None and idx >= args.max_questions:
                break

            question = str(example["question"])
            answers = normalize_answers(example)

            retrieved_passages = retrieve_topk(
                searcher=searcher,
                question=question,
                top_k=args.top_k,
            )

            record = {
                "id": str(example.get("id", idx)),
                "question": question,
                "answers": answers,
                "retrieved_passages": retrieved_passages,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved retrieval results to: {args.output}")


if __name__ == "__main__":
    main()
