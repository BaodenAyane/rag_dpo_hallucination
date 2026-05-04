"""
Build a Pyserini BM25 index over Wiki DPR passages.

Input Wiki DPR passage format:
{
  "id": "wiki_0",
  "source_id": "1",
  "title": "Aaron",
  "text": "Aaron Aaron ..."
}

This script first converts the passages into Pyserini JsonCollection format:
{
  "id": "wiki_0",
  "contents": "Aaron: Aaron Aaron ...",
  "source_id": "1",
  "title": "Aaron",
  "text": "Aaron Aaron ..."
}

Then it builds a Lucene BM25 index using Pyserini.

Example:
  python scripts/01_build_bm25_index.py \
    --passages data/raw/wiki_dpr/wiki_dpr_first_10_shards_text_only.jsonl \
    --index_dir indexes/bm25/wiki_dpr_10_shards \
    --collection_dir data/pyserini/wiki_dpr_10_shards \
    --threads 4 \
    --overwrite
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterator


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream records from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def prepare_pyserini_collection(passages_path: Path, collection_dir: Path) -> int:
    """
    Convert Wiki DPR passage JSONL into Pyserini JsonCollection format.

    Pyserini JsonCollection expects files under a directory, where each line is:
      {"id": "...", "contents": "..."}

    We also keep source_id/title/text in the raw JSON so retrieval output can
    recover passage metadata later.
    """
    collection_dir.mkdir(parents=True, exist_ok=True)
    output_path = collection_dir / "docs.jsonl"

    num_passages = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for idx, passage in enumerate(read_jsonl(passages_path)):
            doc_id = str(passage.get("id", f"doc_{idx}"))
            source_id = passage.get("source_id", passage.get("old_id"))
            title = str(passage.get("title", "")).strip()
            text = str(passage.get("text", "")).strip()

            if title:
                contents = f"{title}: {text}"
            else:
                contents = text

            record = {
                "id": doc_id,
                "contents": contents,
                "source_id": source_id,
                "title": title,
                "text": text,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_passages += 1

    return num_passages


def build_lucene_index(
    collection_dir: Path,
    index_dir: Path,
    threads: int,
    overwrite: bool,
) -> None:
    """Build a Lucene index with Pyserini."""
    if index_dir.exists():
        if overwrite:
            shutil.rmtree(index_dir)
        else:
            raise FileExistsError(
                f"Index directory already exists: {index_dir}\n"
                f"Use --overwrite to rebuild it."
            )

    index_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(collection_dir),
        "--index",
        str(index_dir),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]

    print("Running Pyserini index command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


def save_metadata(
    index_dir: Path,
    collection_dir: Path,
    passages_path: Path,
    num_passages: int,
    threads: int,
) -> None:
    """Save lightweight metadata for reproducibility."""
    metadata = {
        "index_type": "pyserini_lucene_bm25",
        "passages_path": str(passages_path),
        "collection_dir": str(collection_dir),
        "num_passages": num_passages,
        "contents_format": "title: text",
        "threads": threads,
        "store_raw": True,
    }

    index_dir.mkdir(parents=True, exist_ok=True)

    with (index_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--index_dir", type=Path, required=True)
    parser.add_argument(
        "--collection_dir",
        type=Path,
        default=None,
        help=(
            "Directory for temporary Pyserini JsonCollection files. "
            "Default: <index_dir>_collection"
        ),
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing index_dir before rebuilding.",
    )
    args = parser.parse_args()

    collection_dir = args.collection_dir
    if collection_dir is None:
        collection_dir = Path(str(args.index_dir) + "_collection")

    print(f"Passages: {args.passages}")
    print(f"Collection dir: {collection_dir}")
    print(f"Index dir: {args.index_dir}")

    print("Converting passages to Pyserini JsonCollection format...")
    num_passages = prepare_pyserini_collection(
        passages_path=args.passages,
        collection_dir=collection_dir,
    )
    print(f"Prepared {num_passages} passages")

    print("Building Pyserini BM25 index...")
    build_lucene_index(
        collection_dir=collection_dir,
        index_dir=args.index_dir,
        threads=args.threads,
        overwrite=args.overwrite,
    )

    save_metadata(
        index_dir=args.index_dir,
        collection_dir=collection_dir,
        passages_path=args.passages,
        num_passages=num_passages,
        threads=args.threads,
    )

    print(f"Saved Lucene BM25 index to: {args.index_dir}")
    print(f"Saved metadata to: {args.index_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()