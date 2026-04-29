from pathlib import Path
import json

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


RAW_DIR = Path("data/raw")
WIKI_DPR_REPO = "facebook/wiki_dpr"
NQ_OPEN_REPO = "google-research-datasets/nq_open"


def save_jsonl(records, output_path: Path) -> None:
    """Save an iterable of dictionaries as a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in tqdm(records, desc=f"Saving {output_path.name}"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def download_nq_open() -> None:
    """Download NQ-Open and save each split as JSONL."""
    dataset = load_dataset(NQ_OPEN_REPO)

    out_dir = RAW_DIR / "nq_open"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, data in dataset.items():
        save_jsonl(data, out_dir / f"{split}.jsonl")

    print(f"Saved NQ-Open to {out_dir}")


def download_wiki_dpr_shards(num_shards: int = 1) -> None:
    """Download Wiki DPR parquet shards and save text-only passages as JSONL."""
    dfs = []

    for shard_id in range(num_shards):
        filename = f"data/psgs_w100/multiset/train-{shard_id:05d}-of-00157.parquet"
        print(f"Downloading {filename}")

        parquet_path = hf_hub_download(
            repo_id=WIKI_DPR_REPO,
            repo_type="dataset",
            filename=filename,
            local_dir=RAW_DIR / "wiki_dpr_parquet",
            local_dir_use_symlinks=False,
        )

        df = pd.read_parquet(parquet_path)
        df = df.drop(columns=["embeddings"], errors="ignore")
        dfs.append(df)

    wiki_df = pd.concat(dfs, ignore_index=True)
    wiki_df.insert(0, "pid", range(len(wiki_df)))

    if "id" in wiki_df.columns:
        wiki_df = wiki_df.rename(columns={"id": "old_id"})

    wiki_df = wiki_df[["pid", "old_id", "title", "text"]]

    out_dir = RAW_DIR / "wiki_dpr"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"wiki_dpr_first_{num_shards}_shards_text_only.jsonl"
    wiki_df.to_json(out_path, orient="records", lines=True, force_ascii=False)

    print(f"Saved {len(wiki_df)} passages to {out_path}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    download_nq_open()
    download_wiki_dpr_shards(num_shards=1)


if __name__ == "__main__":
    main()