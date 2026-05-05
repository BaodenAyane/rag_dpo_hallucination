# RAG-DPO Hallucination

This project studies whether preference optimization can reduce hallucination in retrieval-augmented generation.

The main experiment uses NQ-Open questions, Wiki DPR passages, Pyserini/Lucene BM25 retrieval, and Qwen2.5-7B-Instruct as the RAG generator. The goal is to train the generator to better distinguish between supported and unsupported retrieval contexts.

## Current Pipeline

1. Download NQ-Open questions and Wiki DPR passages.
2. Build a Pyserini/Lucene BM25 index over Wiki DPR passages.
3. Retrieve top-k passages for NQ-Open questions.
4. Evaluate weak retrieval recall by checking whether any gold answer appears in retrieved passages.
5. Generate baseline RAG answers with Qwen2.5-7B-Instruct.
6. Evaluate baseline generations with EM/F1, citation rate, abstention rate, and retrieved-answer support.
7. Build DPO preference pairs from baseline generation behavior:
   - Supported context + baseline abstention:
     - chosen = answer with evidence citation
     - rejected = insufficient evidence
   - Unsupported context + baseline non-abstention:
     - chosen = insufficient evidence
     - rejected = unsupported baseline answer
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
- [x] DPO pair construction
- [x] DPO training
- [x] DPO model generation
- [x] DPO evaluation
- [x] Baseline vs DPO comparison

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

## DPO Data Construction

DPO data is constructed conservatively to reduce noisy preference pairs.

Supported-answer pairs are built mainly from cases where the retrieved passages contain a gold answer but the baseline model abstained. Unsupported-abstention pairs are built from cases where the retrieved passages do not contain a gold answer but the baseline model still produced an answer.

This conservative construction avoids many noisy cases where NQ exact match marks a semantically acceptable answer as wrong.

Current train DPO data statistics:

```json
{
  "num_examples": 87925,
  "available_supported_pairs": 5587,
  "available_unsupported_pairs": 17745,
  "final_dpo_examples": 11174,
  "final_supported_pairs": 5587,
  "final_unsupported_pairs": 5587
}
```

## Validation Results

Validation set size: 3610 examples.

Detailed results are available in:

```text
results/baseline_vs_dpo_v1.md
```

### Baseline vs DPO-v1

| Metric | Baseline | DPO-v1 | Delta |
|---|---:|---:|---:|
| Exact Match | 0.2582 | 0.2740 | +0.0158 |
| F1 | 0.3322 | 0.3492 | +0.0170 |
| Citation Rate | 0.4102 | 0.8770 | +0.4668 |
| Valid Citation Rate | 0.4102 | 0.8770 | +0.4668 |
| Abstention Rate | 0.3413 | 0.3017 | -0.0396 |
| Retrieved Answer Recall | 0.5623 | 0.5623 | +0.0000 |
| Supported EM | 0.4483 | 0.4685 | +0.0202 |
| Supported F1 | 0.5368 | 0.5552 | +0.0183 |
| Supported Citation Rate | 0.5374 | 0.9552 | +0.4177 |
| Supported Abstention Rate | 0.1709 | 0.1429 | -0.0281 |
| Unsupported EM | 0.0139 | 0.0241 | +0.0101 |
| Unsupported F1 | 0.0694 | 0.0847 | +0.0153 |
| Unsupported Citation Rate | 0.2468 | 0.7766 | +0.5297 |
| Unsupported Abstention Rate | 0.5601 | 0.5057 | -0.0544 |

## Main Finding

DPO-v1 improves answer accuracy and citation formatting, but it does not reduce hallucination under the current weak support labels.

The DPO-tuned model becomes more answer-seeking and citation-seeking:

- Overall EM and F1 improve.
- Supported-context EM and F1 improve.
- Supported-context abstention decreases.
- Citation rate increases substantially.

However:

- Unsupported-context abstention decreases.
- Unsupported-context citation rate increases sharply.
- The model becomes more likely to provide citation-formatted answers even when the retrieved passages do not contain the gold answer.

This suggests that naive weak-label DPO can improve RAG answerability and citation formatting, but may also amplify citation-style hallucination on unsupported examples.

## Interpretation

The current support label is weak:

```text
answer_in_retrieved = whether any gold answer string appears in retrieved passages
```

This does not guarantee that the passage actually supports the question-answer relation. As a result, some DPO pairs may encourage the model to trust retrieved passages too aggressively.

DPO-v1 should therefore be interpreted as an answerability-oriented DPO baseline rather than a fully hallucination-reducing method.

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

### 1. Build DPO Data

```bash
python scripts/06_build_dpo_data.py \
  --eval_file outputs/baseline/base_train_eval_top10_full.jsonl \
  --output_file data/preference/dpo_train_top10_full.jsonl \
  --stats_file outputs/baseline/dpo_train_stats_top10_full.json \
  --supported_ratio 0.5 \
  --supported_mode abstention_only \
  --max_unsupported_per_supported 1.0 \
  --seed 42
```

### 2. Train DPO-v1

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1 python -u scripts/07_train_dpo.py \
  --train_file data/preference/dpo_train_top10_full.jsonl \
  --eval_file data/preference/dpo_val_top10_full.jsonl \
  --output_dir outputs/dpo/qwen2_5_7b_rag_dpo_top10_full_1epoch_len3328_lr1e6 \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-6 \
  --num_train_epochs 1 \
  --max_prompt_length 3072 \
  --max_length 3328 \
  --beta 0.1 \
  --lora_r 16 \
  --lora_alpha 32 \
  --bf16 \
  --save_steps 100 \
  --eval_steps 250 \
  --logging_steps 10 \
  --eval_strategy steps \
  --overwrite_output_dir
```

### 3. Generate with DPO Adapter

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/08_generate_dpo.py \
  --retrieval_file data/retrieval/nq_validation_bm25_top10_full.jsonl \
  --output data/generation/dpo_val_outputs_top10_full.jsonl \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --adapter_dir outputs/dpo/qwen2_5_7b_rag_dpo_top10_full_1epoch_len3328_lr1e6 \
  --top_k 10 \
  --max_new_tokens 64 \
  --max_input_length 4096 \
  --batch_size 4 \
  --resume
```

### 4. Evaluate DPO Generations

```bash
python scripts/05_eval_generation.py \
  --generation_file data/generation/dpo_val_outputs_top10_full.jsonl \
  --output_file outputs/dpo/dpo_val_metrics_top10_full.json \
  --per_example_output outputs/dpo/dpo_val_eval_top10_full.jsonl
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
    baseline_vs_dpo_v1.md
    baseline_val_metrics_top10_full.json
    dpo_v1_val_metrics_top10_full.json
    dpo_v1_data_stats.json

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

The LoRA adapter checkpoint is not included in the repository. It should be stored separately, for example on Hugging Face Hub or cloud storage.

## Future Work

- Add verifier-based filtering for supported preference pairs.
- Use an LLM or NLI model to check whether passages actually support the chosen answer.
- Increase the ratio of unsupported-abstention pairs.
- Add stronger “no citation when unsupported” preference examples.
- Evaluate citation faithfulness beyond citation format validity.
- Add list-aware evaluation for multi-answer NQ examples.
- Compare DPO-only, SFT-only, and SFT+DPO variants.