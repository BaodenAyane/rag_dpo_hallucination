# RAG-DPO Hallucination

This project studies whether preference optimization can reduce hallucination in retrieval-augmented generation.

The main experiment uses NQ-Open questions, Wiki DPR passages, Pyserini BM25 retrieval, and Qwen2.5-7B-Instruct as the RAG generator. The goal is to study whether weak-label Direct Preference Optimization (DPO) can improve evidence-grounded answering and reduce hallucination in RAG.

The main finding is cautionary: weak-label DPO produces only modest answer-quality gains, while naive DPO can substantially amplify unfaithful citation behavior. A cleaner DPO variant mitigates citation misuse, but does not outperform the original baseline on overall hallucination-related metrics.

## Current Pipeline

1. Download NQ-Open questions and Wiki DPR passages.
2. Build a Pyserini/Lucene BM25 index over Wiki DPR passages.
3. Retrieve top-k passages for NQ-Open questions.
4. Evaluate weak retrieval recall by checking whether any gold answer appears in retrieved passages.
5. Generate baseline RAG answers with Qwen2.5-7B-Instruct.
6. Evaluate baseline generations with EM/F1, citation rate, abstention rate, and weak retrieved-answer support.
7. Build DPO preference pairs from baseline generation behavior.
8. Train the generator with LoRA DPO.
9. Generate answers with the DPO-tuned model.
10. Compare baseline vs DPO variants using automatic metrics.
11. Run an LLM verifier audit to evaluate answer correctness, passage support, citation faithfulness, and hallucination.

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
- [x] Baseline vs DPO-v1 vs DPO-v2 automatic comparison
- [x] Full-validation LLM verifier audit
- [x] Final analysis

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
- LLM verifier: external OpenAI-compatible judge API

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

### Weak Support Definition

The project uses a weak automatic support proxy:

```text
answer_in_retrieved = whether any gold answer string appears in the retrieved passages
```

This is not a human-verified support label. Therefore, the supported/unsupported split should be interpreted as:

```text
gold-in-retrieved
gold-not-in-retrieved
```

rather than strict evidence-supported and evidence-unsupported labels.

This limitation is important because some examples marked as unsupported may still contain semantically useful evidence, aliases, partially relevant evidence, or noisy gold labels.

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

## Automatic Validation Results

Validation set size: 3610 examples.

Detailed automatic results are available in:

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

### Notes on Automatic Metrics

The automatic evaluation is useful for measuring EM/F1, abstention behavior, and citation-format behavior. However, it has two important limitations:

1. `valid_citation_rate` only checks whether citation IDs refer to retrieved passages. It does not check whether the cited passages actually support the answer.
2. The supported/unsupported split is based on gold-answer string matching, not human-verified evidence support.

Therefore, automatic unsupported EM/F1 should be interpreted cautiously. Higher unsupported EM/F1 may indicate that the model answers more often or uses parametric knowledge, rather than that its answers are grounded in retrieved evidence.

## LLM Verifier Audit

To evaluate grounding more directly, this project also runs a full-validation LLM verifier audit.

The verifier judges each system output using the question, gold answers, retrieved passages, and generated answer. It labels:

- answer type
- gold correctness
- passage support
- citation faithfulness
- hallucination

The verifier is instructed to judge passage support only from the retrieved passages and not to use outside knowledge for evidence support.

### LLM Verifier Results

| Metric | Baseline | DPO-v1 | DPO-v2-lightclean |
|---|---:|---:|---:|
| Answer Rate | 65.82% | 69.83% | 69.72% |
| Abstention Rate | 34.13% | 30.17% | 30.25% |
| Gold Correct / Partial | 42.27% | 44.24% | 43.79% |
| Passage Supported / Partial | 51.91% | 52.44% | 53.13% |
| Unfaithful Citation Rate | 9.45% | 37.70% | 17.29% |
| No Citation Rate | 58.64% | 12.24% | 41.41% |
| Hallucination / Partial | 16.68% | 19.70% | 19.61% |
| Non-Hallucination Rate | 82.74% | 80.19% | 80.22% |

### LLM Verifier Findings

The LLM verifier audit shows that DPO-v1 improves answer-seeking behavior, but introduces substantial citation misuse.

DPO-v1 raises gold-correct-or-partial rate from 42.27% to 44.24%, but also increases unfaithful citation rate from 9.45% to 37.70%.

DPO-v2-lightclean reduces the DPO-v1 unfaithful citation rate from 37.70% to 17.29%, a relative reduction of more than 50%. It also preserves most of the answer-quality improvement from DPO-v1.

However, DPO-v2-lightclean does not outperform the baseline on overall hallucination-related metrics. The verifier-labeled hallucination-or-partial rate remains higher than the baseline.

## Main Findings

1. Baseline remains the strongest system for hallucination calibration and citation faithfulness.

   The baseline has the lowest unfaithful citation rate and the lowest verifier-labeled hallucination-or-partial rate.

2. DPO-v1 produces only modest answer-quality gains.

   DPO-v1 has the highest automatic EM/F1 and slightly improves LLM-judged gold correctness, but the improvement is small.

3. DPO-v1 severely amplifies unfaithful citation.

   DPO-v1 dramatically increases citation frequency and unfaithful citation rate. This shows that higher citation rate or valid citation ID rate does not imply faithful citation.

4. DPO-v2-lightclean mitigates citation misuse.

   DPO-v2-lightclean substantially reduces DPO-v1's unfaithful citation problem while preserving most of the supported-answer gains.

5. Weak-label DPO is not sufficient for robust hallucination reduction.

   The weak preference labels are noisy because they rely on gold-answer string matching rather than human-verified or verifier-verified evidence support.

## Interpretation

Naive weak-label DPO makes the model more answer-seeking and citation-seeking. This can slightly improve answer accuracy, but it also harms citation faithfulness and hallucination calibration.

DPO-v2-lightclean shows that data cleaning and no-citation instructions can reduce one major failure mode: unfaithful citation. However, it does not fully solve answer-level hallucination.

The results suggest that DPO is not automatically hallucination-reducing in RAG. For hallucination reduction, preference data quality is more important than simply optimizing on weak preference pairs.

## Takeaway

Weak-label DPO improves answerability only modestly and can amplify citation-style hallucination.

Cleaner DPO construction reduces citation misuse, but the baseline remains better on overall hallucination and citation-faithfulness metrics.

Future DPO-based hallucination reduction likely requires higher-quality preference labels, such as human annotation, LLM-verifier labels, or NLI-based support verification.

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
scripts/09_llm_verifier_audit.py
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

### Run LLM Verifier Audit

Set API credentials first:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="your_openai_compatible_base_url"
```

Run full validation audit:

```bash
python scripts/09_llm_verifier_audit.py \
  --baseline_eval outputs/baseline/base_val_eval_top10_full.jsonl \
  --dpo_v1_eval outputs/dpo/dpo_val_eval_top10_full.jsonl \
  --dpo_v2_eval outputs/dpo/dpo_v2_u2_lightclean_val_eval_top10_full.jsonl \
  --output_file results/llm_verifier_audit_val_all_ecnu_v2.jsonl \
  --summary_file results/llm_verifier_audit_val_all_ecnu_v2_summary.json \
  --sample_size 0 \
  --split all \
  --model ecnu-plus \
  --base_url https://chat.ecnu.edu.cn/open/api/v1 \
  --resume
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
    09_llm_verifier_audit.py

  results/
    baseline_vs_dpo_v1_v2.md
    baseline_val_metrics_top10_full.json
    dpo_v1_val_metrics_top10_full.json
    dpo_v2_lightclean_val_metrics_top10_full.json
    llm_verifier_audit_val_all_ecnu_v2_summary.json

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

API keys and local environment files should not be committed:

```text
.env
*.env
```

## Limitations

- The preference data is constructed from weak automatic labels, not human annotations.
- The automatic supported/unsupported split is based on gold-answer string matching.
- Automatic citation validity only checks citation IDs, not citation faithfulness.
- The LLM verifier audit is stronger than string matching, but still depends on the judge model.
- The generator is a 7B instruct model, which may limit long-context citation and abstention behavior.
- The project does not include human-labeled preference data.

## Future Work

- Run prompt ablation with stricter citation rules:
  - If the answer is insufficient evidence, do not cite any passage.
- Build verifier-labeled preference data instead of weak string-match labels.
- Use an LLM or NLI model to check whether passages actually support the chosen answer.
- Evaluate citation faithfulness beyond citation format validity.
- Add list-aware evaluation for multi-answer NQ examples.
- Compare DPO-only, SFT-only, and SFT+DPO variants.
- Test stronger base models with better RAG and citation-following ability.
