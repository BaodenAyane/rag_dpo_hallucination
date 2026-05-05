# RAG-DPO Hallucination

This project studies whether preference optimization can reduce hallucination in retrieval-augmented generation.

The main experiment uses NQ-Open questions, Wiki DPR passages, Pyserini BM25 retrieval, and Qwen2.5-7B-Instruct as the RAG generator. The goal is to train the generator to better distinguish between supported and unsupported retrieval contexts.

## Current Pipeline

1. Download NQ-Open questions and Wiki DPR passages.
2. Build a Pyserini/Lucene BM25 index over Wiki DPR passages.
3. Retrieve top-k passages for NQ-Open questions.
4. Evaluate weak retrieval recall by checking whether any gold answer appears in retrieved passages.
5. Generate baseline RAG answers with Qwen2.5-7B-Instruct.
6. Evaluate baseline generations with EM/F1, citation rate, abstention rate, and retrieved-answer support.
7. Build DPO preference pairs from baseline generation behavior.
8. Train the generator with LoRA DPO.
9. Generate answers with the DPO-tuned model.
10. Compare baseline vs DPO on validation metrics.

## Status

- [x] Data download
- [x] Wiki DPR full-corpus preprocessing
- [x] Pyserini/Lucene BM25 index construction
- [x] BM25 retrieval
- [x] Retrieval recall evaluation
- [x] Baseline RAG generation
- [x] Baseline generation evaluation
- [x] Evidence support filtering
- [x] DPO-v1 pair construction
- [x] DPO-v1 training
- [x] DPO-v1 generation
- [x] DPO-v1 evaluation
- [x] DPO-v2-lightclean pair construction
- [x] DPO-v2-lightclean training
- [x] DPO-v2-lightclean generation
- [x] DPO-v2-lightclean evaluation
- [x] Baseline vs DPO-v1 vs DPO-v2 comparison

## Experimental Setup

- Dataset: NQ-Open
- Corpus: Wiki DPR passages
- Retriever: Pyserini/Lucene BM25
- Generator: Qwen/Qwen2.5-7B-Instruct
- Retrieval top-k for generation: 10
- Generation max new tokens: 64
- Generation max input length: 4096
- DPO method: LoRA DPO
- DPO data source: baseline RAG generation outputs

## DPO Training Setup

### DPO-v1

- Base model: Qwen/Qwen2.5-7B-Instruct
- Fine-tuning method: LoRA DPO
- LoRA rank: 16
- LoRA alpha: 32
- DPO beta: 0.1
- Learning rate: 1e-6
- Epochs: 1
- Max prompt length: 3072
- Max sequence length: 3328
- Supported / unsupported pair ratio: 1:1

### DPO-v2-lightclean

- Base model: Qwen/Qwen2.5-7B-Instruct
- Fine-tuning method: LoRA DPO
- LoRA rank: 16
- LoRA alpha: 32
- DPO beta: 0.1
- Learning rate: 5e-7
- Epochs: 1
- Max prompt length: 3072
- Max sequence length: 3328
- Final supported / unsupported pair ratio: approximately 1:2
- Training: 2-GPU DDP data parallelism

## DPO Data Construction

### DPO-v1

DPO-v1 uses weak preference pairs constructed from baseline RAG behavior.

Supported-answer pairs:

- Condition: retrieved passages contain a gold answer, but the baseline model abstained.
- chosen = gold answer with evidence citation
- rejected = baseline insufficient-evidence response

Unsupported-abstention pairs:

- Condition: retrieved passages do not contain a gold answer, but the baseline model answered.
- chosen = insufficient evidence
- rejected = unsupported baseline answer

DPO-v1 train data statistics:

```json
{
  "final_dpo_examples": 11174,
  "final_supported_pairs": 5587,
  "final_unsupported_pairs": 5587
}
```

### DPO-v2-lightclean

DPO-v2-lightclean is designed to reduce the unsupported citation explosion observed in DPO-v1.

It modifies the data construction in three ways:

1. Uses an unsupported-heavy final ratio of approximately 1:2.
2. Adds explicit no-citation wording to unsupported chosen responses.
3. Filters unsupported examples where the baseline short answer itself appears to be supported by retrieved passages.

DPO-v2-lightclean train data statistics:

```json
{
  "num_examples": 87925,
  "available_supported_pairs": 6585,
  "available_unsupported_pairs": 4077,
  "final_dpo_examples": 6112,
  "final_supported_pairs": 2035,
  "final_unsupported_pairs": 4077,
  "unsupported_baseline_answer_in_passages_skipped": 11934
}
```

## Validation Results

Validation set size: 3610 examples.

Detailed results are available in:

```text
results/baseline_vs_dpo_v1_v2.md
```

### Baseline vs DPO-v1 vs DPO-v2-lightclean

| Metric | Baseline | DPO-v1 | DPO-v2-lightclean |
|---|---:|---:|---:|
| Exact Match | 0.2582 | 0.2740 | 0.2684 |
| F1 | 0.3322 | 0.3492 | 0.3447 |
| Citation Rate | 0.4102 | 0.8770 | 0.5856 |
| Valid Citation Rate | 0.4102 | 0.8770 | 0.5856 |
| Abstention Rate | 0.3413 | 0.3017 | 0.3025 |
| Retrieved Answer Recall | 0.5623 | 0.5623 | 0.5623 |
| Supported EM | 0.4483 | 0.4685 | 0.4655 |
| Supported F1 | 0.5368 | 0.5552 | 0.5543 |
| Supported Citation Rate | 0.5374 | 0.9552 | 0.7281 |
| Supported Abstention Rate | 0.1709 | 0.1429 | 0.1374 |
| Unsupported EM | 0.0139 | 0.0241 | 0.0152 |
| Unsupported F1 | 0.0694 | 0.0847 | 0.0755 |
| Unsupported Citation Rate | 0.2468 | 0.7766 | 0.4025 |
| Unsupported Abstention Rate | 0.5601 | 0.5057 | 0.5146 |

## Main Findings

DPO-v1 improves answer accuracy and citation formatting, but it also sharply increases citation behavior on unsupported examples.

DPO-v2-lightclean preserves most of the supported-answer gains from DPO-v1 while substantially reducing unsupported citation rate.

Key observations:

- DPO-v1 has the highest overall EM/F1.
- DPO-v1 causes unsupported citation rate to rise from 0.2468 to 0.7766.
- DPO-v2-lightclean reduces unsupported citation rate from 0.7766 to 0.4025.
- DPO-v2-lightclean keeps supported F1 close to DPO-v1.
- Neither DPO-v1 nor DPO-v2-lightclean restores unsupported abstention rate to the baseline level.

## Interpretation

DPO-v1 appears to make the model more answer-seeking and citation-seeking. This improves supported-answer performance, but it also encourages citation-formatted answers in unsupported contexts.

DPO-v2-lightclean partially fixes this by using cleaner unsupported pairs and explicit no-citation chosen responses. It significantly reduces unsupported citation rate while preserving most supported-answer gains.

However, unsupported abstention remains below the baseline. This suggests that weak-label DPO alone is not sufficient to fully teach robust evidence-grounded abstention.

## Takeaway

DPO can improve RAG answerability and citation behavior, but naive weak-label DPO may amplify citation-style hallucination.

A cleaner DPO construction reduces this effect, but further work is needed to improve abstention behavior on unsupported retrieval contexts.

## Scripts

```text
scripts/00_download_data.py
scripts/01_build_bm25_index.py
scripts/02_retrieve_bm25.py
scripts/03_eval_retrieval.py
scripts/04_generate_baseline.py
scripts/05_eval_generation.py
scripts/06_build_dpo_data.py
scripts/07_train_dpo.py
scripts/08_generate_dpo.py
```

## Reproduction Commands

### Build DPO-v2-lightclean Data

```bash
python scripts/06_build_dpo_data.py \
  --eval_file outputs/baseline/base_train_eval_top10_full.jsonl \
  --output_file data/preference/dpo_train_top10_full_u2_lightclean.jsonl \
  --stats_file outputs/baseline/dpo_train_stats_top10_full_u2_lightclean.json \
  --supported_ratio 0.333 \
  --supported_mode abstention_only \
  --max_unsupported_per_supported 2.0 \
  --seed 42
```

Validation preference data:

```bash
python scripts/06_build_dpo_data.py \
  --eval_file outputs/baseline/base_val_eval_top10_full.jsonl \
  --output_file data/preference/dpo_val_top10_full_u2_lightclean.jsonl \
  --stats_file outputs/baseline/dpo_val_stats_top10_full_u2_lightclean.json \
  --supported_ratio 0.333 \
  --supported_mode abstention_only \
  --max_unsupported_per_supported 2.0 \
  --seed 42
```

### Train DPO-v2-lightclean

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/07_train_dpo.py \
  --train_file data/preference/dpo_train_top10_full_u2_lightclean.jsonl \
  --eval_file data/preference/dpo_val_top10_full_u2_lightclean.jsonl \
  --output_dir outputs/dpo/qwen2_5_7b_rag_dpo_v2_u2_lightclean_len3328_lr5e7_ddp2 \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-7 \
  --num_train_epochs 1 \
  --max_prompt_length 3072 \
  --max_length 3328 \
  --beta 0.1 \
  --lora_r 16 \
  --lora_alpha 32 \
  --bf16 \
  --save_steps 100 \
  --eval_steps 100 \
  --logging_steps 10 \
  --eval_strategy steps \
  --overwrite_output_dir
```

### Generate with DPO-v2-lightclean Adapter

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/08_generate_dpo.py \
  --retrieval_file data/retrieval/nq_validation_bm25_top10_full.jsonl \
  --output data/generation/dpo_v2_u2_lightclean_val_outputs_top10_full.jsonl \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --adapter_dir outputs/dpo/qwen2_5_7b_rag_dpo_v2_u2_lightclean_len3328_lr5e7_ddp2 \
  --top_k 10 \
  --max_new_tokens 64 \
  --max_input_length 4096 \
  --batch_size 4 \
  --resume
```

### Evaluate DPO-v2-lightclean Generations

```bash
python scripts/05_eval_generation.py \
  --generation_file data/generation/dpo_v2_u2_lightclean_val_outputs_top10_full.jsonl \
  --output_file outputs/dpo/dpo_v2_u2_lightclean_val_metrics_top10_full.json \
  --per_example_output outputs/dpo/dpo_v2_u2_lightclean_val_eval_top10_full.jsonl
```

## Repository Contents

Recommended repository structure:

```text
rag_dpo_hallucination/
  README.md
  requirements.txt
  .gitignore

  scripts/
    00_download_data.py
    01_build_bm25_index.py
    02_retrieve_bm25.py
    03_eval_retrieval.py
    04_generate_baseline.py
    05_eval_generation.py
    06_build_dpo_data.py
    07_train_dpo.py
    08_generate_dpo.py

  results/
    baseline_vs_dpo_v1_v2.md
    baseline_val_metrics_top10_full.json
    dpo_v1_val_metrics_top10_full.json
    dpo_v2_lightclean_val_metrics_top10_full.json

  examples/
    dpo_supported_examples.jsonl
    dpo_unsupported_examples.jsonl
```

## Files Not Included

Large generated artifacts should not be committed to GitHub:

```text
data/raw/
data/pyserini/
data/retrieval/
data/generation/
data/preference/
indexes/
outputs/dpo/
logs/
```

The LoRA adapter checkpoints are not included in the repository. They should be stored separately, for example on Hugging Face Hub or cloud storage.

## Future Work

- Run prompt ablation with stricter citation rules:
  - If the answer is insufficient evidence, do not cite any passage.
- Add verifier-based filtering for supported and unsupported preference pairs.
- Use an LLM or NLI model to check whether passages actually support the chosen answer.
- Evaluate citation faithfulness beyond citation format validity.
- Add list-aware evaluation for multi-answer NQ examples.
- Compare DPO-only, SFT-only, and SFT+DPO variants.
