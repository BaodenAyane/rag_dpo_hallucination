"""
Download NQ-Open and Wiki DPR passage shards for the RAG baseline.

Outputs:
  data/raw/nq_open/train.jsonl
  data/raw/nq_open/validation.jsonl
  data/raw/wiki_dpr/wiki_dpr_first_{num_shards}_shards_text_only.jsonl

Example:
  python scripts/00_download_data.py \
    --num_wiki_shards 157 \
    --batch_size 10000 \
    --skip_nq

Small debug example:
  python scripts/00_download_data.py \
    --num_wiki_shards 1 \
    --batch_size 5000
"""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


RAW_DIR = Path("data/raw")
WIKI_DPR_REPO = "facebook/wiki_dpr"
NQ_OPEN_REPO = "google-research-datasets/nq_open"


def save_jsonl(records: Iterable[Dict[str, Any]], output_path: Path) -> None:
    """Save dictionaries to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in tqdm(records, desc=f"Saving {output_path.name}"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_nq_record(record: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Normalize an NQ-Open example to the project schema."""
    answers = record.get("answer", record.get("answers", []))

    if isinstance(answers, str):
        answers = [answers]

    if answers is None:
        answers = []

    return {
        "id": str(record.get("id", idx)),
        "question": record["question"],
        "answers": [str(a) for a in answers],
    }


def download_nq_open() -> None:
    """Download NQ-Open and save each split as JSONL."""
    dataset = load_dataset(NQ_OPEN_REPO)
    out_dir = RAW_DIR / "nq_open"

    for split, data in dataset.items():
        records = (
            format_nq_record(record, idx)
            for idx, record in enumerate(data)
        )
        save_jsonl(records, out_dir / f"{split}.jsonl")

    print(f"Saved NQ-Open to {out_dir}")


def hf_download_with_retry(
    repo_id: str,
    repo_type: str,
    filename: str,
    local_dir: Path,
    max_retries: int,
    sleep_seconds: int,
) -> str:
    """Download a Hugging Face file with simple retry logic."""
    local_dir.mkdir(parents=True, exist_ok=True)
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {filename} attempt {attempt}/{max_retries}")
            return hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=filename,
                local_dir=local_dir,
            )
        except Exception as exc:
            last_error = exc
            print(f"Download failed for {filename}: {repr(exc)}")

            if attempt < max_retries:
                print(f"Retrying in {sleep_seconds} seconds...")
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to download {filename}") from last_error


def safe_get(values, i: int, default: str = "") -> str:
    """Safely read one value from a list-like object."""
    try:
        value = values[i]
    except Exception:
        return default

    if value is None:
        return default

    return str(value)


def convert_parquet_to_jsonl_append(
    parquet_path: str,
    fout,
    start_idx: int,
    batch_size: int,
) -> int:
    """
    Stream-convert one Wiki DPR parquet shard to JSONL.

    Only id/title/text columns are read; embeddings are never loaded.
    Returns the next global wiki index.
    """
    parquet_file = pq.ParquetFile(parquet_path)

    global_idx = start_idx

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["id", "title", "text"],
    ):
        batch_dict = batch.to_pydict()
        num_rows = batch.num_rows

        ids = batch_dict.get("id", [""] * num_rows)
        titles = batch_dict.get("title", [""] * num_rows)
        texts = batch_dict.get("text", [""] * num_rows)

        for i in range(num_rows):
            record = {
                "id": f"wiki_{global_idx}",
                "source_id": safe_get(ids, i),
                "title": safe_get(titles, i),
                "text": safe_get(texts, i),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            global_idx += 1

        del batch
        del batch_dict
        del ids
        del titles
        del texts
        gc.collect()

    del parquet_file
    gc.collect()

    return global_idx


def download_wiki_dpr_shards(
    num_shards: int = 1,
    batch_size: int = 10000,
    max_retries: int = 5,
    sleep_seconds: int = 20,
) -> None:
    """
    Download Wiki DPR parquet shards and save text-only passages as JSONL.

    Streaming version:
    - Reads only id/title/text columns.
    - Does not load embeddings.
    - Does not concatenate DataFrames.
    - Writes JSONL incrementally.
    """
    if num_shards < 1 or num_shards > 157:
        raise ValueError("--num_wiki_shards must be between 1 and 157.")

    out_dir = RAW_DIR / "wiki_dpr"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"wiki_dpr_first_{num_shards}_shards_text_only.jsonl"

    parquet_dir = RAW_DIR / "wiki_dpr_parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    global_idx = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for shard_id in range(num_shards):
            filename = f"data/psgs_w100/multiset/train-{shard_id:05d}-of-00157.parquet"

            parquet_path = hf_download_with_retry(
                repo_id=WIKI_DPR_REPO,
                repo_type="dataset",
                filename=filename,
                local_dir=parquet_dir,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
            )

            print(f"Converting {parquet_path}")

            before = global_idx
            global_idx = convert_parquet_to_jsonl_append(
                parquet_path=parquet_path,
                fout=fout,
                start_idx=global_idx,
                batch_size=batch_size,
            )
            fout.flush()

            print(
                f"Finished shard {shard_id + 1}/{num_shards}; "
                f"added {global_idx - before} passages; "
                f"total passages so far: {global_idx}"
            )

            gc.collect()

    print(f"Saved {global_idx} passages to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_wiki_shards", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--skip_nq", action="store_true")
    parser.add_argument("--skip_wiki", action="store_true")
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--sleep_seconds", type=int, default=20)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_nq:
        download_nq_open()

    if not args.skip_wiki:
        download_wiki_dpr_shards(
            num_shards=args.num_wiki_shards,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
        )


if __name__ == "__main__":
    main()