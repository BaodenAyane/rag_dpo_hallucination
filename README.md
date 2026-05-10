# RAG-DPO Hallucination

This project studies hallucination reduction in retrieval-augmented question answering (RAG) using weak-label Direct Preference Optimization (DPO) and inference-time self-verification.

The main experimental setting uses NQ-Open questions, Wiki DPR passages, Pyserini BM25 retrieval, and Qwen2.5-7B-Instruct as the RAG generator. The goal is to examine whether preference optimization can improve evidence-grounded answering and reduce unsupported generation under retrieval uncertainty.

The main finding is cautionary: weak-label DPO modestly improves answer accuracy when retrieved evidence is sufficient, but it can also make the model more answer-seeking and citation-seeking, increasing overconfident answers and unfaithful citations when retrieval evidence is insufficient. An inference-time SelfCheck verifier reduces hallucination-style behavior, but introduces a safety–coverage trade-off by increasing over-refusal.

## Research Questions

This project investigates three questions:

1. Can weak-label DPO improve evidence-grounded answering in RAG?
2. Does DPO reduce hallucination when retrieval evidence is insufficient?
3. Can inference-time self-verification reduce unsupported answers and citation misuse?

## Main Contributions

* Build a full RAG pipeline over NQ-Open and Wiki DPR using BM25 retrieval and Qwen2.5-7B-Instruct generation.
* Construct weak DPO preference pairs from baseline RAG behavior.
* Train and compare two LoRA-DPO variants:

  * DPO-v1: balanced supported/unsupported weak preference data.
  * DPO-v2-lightclean: cleaned and unsupported-heavy preference data designed to reduce citation misuse.
* Propose a RAG-behavior evaluation that distinguishes:

  * evidence-sufficient examples where the model should answer;
  * evidence-insufficient examples where the model should abstain.
* Evaluate inference-time SelfCheck as an output verification guardrail.
* Run an LLM verifier audit for answer correctness, passage support, citation faithfulness, and hallucination.

## High-Level Findings

1. **DPO improves answer-seeking behavior under sufficient evidence.**
   DPO-v1 and DPO-v2-lightclean improve sufficient-evidence EM/F1 and reduce over-refusal compared with the baseline.

2. **Naive DPO can worsen behavior under insufficient evidence.**
   DPO-v1 answers more often even when retrieved passages do not contain sufficient evidence, increasing overconfident answering and citation misuse.

3. **Citation rate is not citation faithfulness.**
   DPO-v1 greatly increases citation frequency, but the LLM verifier shows that many citations are unfaithful.

4. **Light cleaning reduces citation misuse but does not fully solve hallucination.**
   DPO-v2-lightclean substantially reduces the unfaithful citation problem introduced by DPO-v1, while preserving most answer-quality gains.

5. **Inference-time SelfCheck improves safety but increases over-refusal.**
   SelfCheck reduces hallucination-style bad answer behavior, especially on evidence-insufficient examples, but it also rejects some answerable cases.

## Current Pipeline

1. Download NQ-Open questions and Wiki DPR passages.
2. Build a Pyserini/Lucene BM25 index over Wiki DPR passages.
3. Retrieve top-k passages for NQ-Open questions.
4. Evaluate weak retrieval recall by checking whether any gold answer appears in retrieved passages.
5. Generate baseline RAG answers with Qwen2.5-7B-Instruct.
6. Evaluate baseline generations with EM/F1, abstention behavior, citation behavior, and RAG-behavior metrics.
7. Build weak-label DPO preference pairs from baseline generation behavior.
8. Train LoRA-DPO adapters.
9. Generate answers with DPO-tuned models.
10. Compare baseline, DPO variants, and SelfCheck using automatic metrics.
11. Run an LLM verifier audit to evaluate answer correctness, passage support, citation faithfulness, and hallucination.

## Status

* [x] Data download
* [x] Wiki DPR full-corpus preprocessing
* [x] Pyserini/Lucene BM25 index construction
* [x] BM25 retrieval
* [x] Retrieval recall evaluation
* [x] Baseline RAG generation
* [x] Baseline generation evaluation
* [x] Evidence support filtering
* [x] DPO-v1 pair construction
* [x] DPO-v1 training
* [x] DPO-v1 generation
* [x] DPO-v1 evaluation
* [x] DPO-v2-lightclean pair construction
* [x] DPO-v2-lightclean training
* [x] DPO-v2-lightclean generation
* [x] DPO-v2-lightclean evaluation
* [x] Inference-time SelfCheck evaluation
* [x] Baseline vs DPO-v1 vs DPO-v2 vs SelfCheck automatic comparison
* [x] Full-validation LLM verifier audit
* [x] Final analysis

## Experimental Setup

* Dataset: NQ-Open
* Corpus: Wiki DPR passages
* Retriever: Pyserini/Lucene BM25
* Generator: Qwen/Qwen2.5-7B-Instruct
* Retrieval top-k for generation: 10
* Generation max new tokens: 64
* Generation max input length: 4096
* DPO method: LoRA DPO
* DPO data source: baseline RAG generation outputs
* SelfCheck method: inference-time output verification
* LLM verifier: external OpenAI-compatible judge API

## DPO Training Setup

### DPO-v1

* Base model: Qwen/Qwen2.5-7B-Instruct
* Fine-tuning method: LoRA DPO
* LoRA rank: 16
* LoRA alpha: 32
* DPO beta: 0.1
* Learning rate: 1e-6
* Epochs: 1
* Max prompt length: 3072
* Max sequence length: 3328
* Supported / unsupported pair ratio: 1:1

### DPO-v2-lightclean

* Base model: Qwen/Qwen2.5-7B-Instruct
* Fine-tuning method: LoRA DPO
* LoRA rank: 16
* LoRA alpha: 32
* DPO beta: 0.1
* Learning rate: 5e-7
* Epochs: 1
* Max prompt length: 3072
* Max sequence length: 3328
* Final supported / unsupported pair ratio: approximately 1:2
* Training: 2-GPU DDP data parallelism

## DPO Data Construction

### Weak Support Definition

The project uses a weak automatic support proxy:

```text
answer_in_retrieved = whether any gold answer string appears in the retrieved passages
```

This is not a human-verified evidence support label. Therefore, the supported/unsupported split should be interpreted as:

```text
gold-in-retrieved
gold-not-in-retrieved
```

rather than strict evidence-supported and evidence-unsupported labels.

This limitation is important because some examples marked as unsupported may still contain semantically useful evidence, aliases, partially relevant evidence, or noisy gold labels.

### DPO-v1

DPO-v1 uses weak preference pairs constructed from baseline RAG behavior.

Supported-answer pairs:

* Condition: retrieved passages contain a gold answer, but the baseline model abstained.
* chosen = gold answer with evidence citation
* rejected = baseline insufficient-evidence response

Unsupported-abstention pairs:

* Condition: retrieved passages do not contain a gold answer, but the baseline model answered.
* chosen = insufficient evidence
* rejected = unsupported baseline answer

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

## Evaluation Protocol

The project reports two kinds of automatic metrics.

### Standard QA and citation diagnostics

These include:

* Exact Match
* F1
* Citation Rate
* Valid Citation Rate
* Abstention Rate
* Retrieved Answer Recall

These metrics are useful diagnostics, but they are not sufficient for evaluating hallucination in RAG.

### RAG-behavior metrics

The main automatic evaluation separates examples into two groups:

```text
evidence_sufficient: retrieved passages contain a gold answer string
evidence_insufficient: retrieved passages do not contain a gold answer string
```

The desired model behavior is:

```text
If evidence_sufficient: answer correctly.
If evidence_insufficient: abstain.
```

The evaluator assigns one of five response types:

| Evidence status | Model behavior                 | Response type        |
| --------------- | ------------------------------ | -------------------- |
| Sufficient      | Correct answer                 | supported_answer     |
| Sufficient      | Incorrect / unsupported answer | unsupported_answer   |
| Sufficient      | Refusal                        | over_refusal         |
| Insufficient    | Refusal                        | correct_refusal      |
| Insufficient    | Concrete answer                | overconfident_answer |

Main behavior metrics:

* `rag_behavior_accuracy`: supported_answer + correct_refusal
* `policy_error_rate`: unsupported_answer + over_refusal + overconfident_answer
* `bad_answer_behavior_rate`: unsupported_answer + overconfident_answer
* `sufficient_supported_answer_rate`
* `sufficient_over_refusal_rate`
* `insufficient_correct_refusal_rate`
* `insufficient_overconfident_answer_rate`

## Automatic Validation Results

Validation set size: 3610 examples.

### RAG-Behavior Results

| Metric                              | Baseline |     DPO-v1 | DPO-v2-lightclean | DPO-v1 + SelfCheck |
| ----------------------------------- | -------: | ---------: | ----------------: | -----------------: |
| RAG Behavior Accuracy ↑             |   0.5429 |     0.5299 |            0.5343 |         **0.5676** |
| Policy Error Rate ↓                 |   0.4571 |     0.4701 |            0.4657 |         **0.4324** |
| Bad Answer Behavior ↓               |   0.3609 |     0.3898 |            0.3884 |         **0.2767** |
| Sufficient Supported Answer ↑       |   0.5296 | **0.5488** |        **0.5498** |             0.4867 |
| Sufficient Unsupported Answer ↓     |   0.2995 |     0.3084 |            0.3128 |         **0.2365** |
| Sufficient Over-refusal ↓           |   0.1709 |     0.1429 |        **0.1374** |             0.2768 |
| Insufficient Correct Refusal ↑      |   0.5601 |     0.5057 |            0.5146 |         **0.6715** |
| Insufficient Overconfident Answer ↓ |   0.4399 |     0.4943 |            0.4854 |         **0.3285** |
| Overall Citation Rate               |   0.4102 |     0.8770 |            0.5856 |             0.5582 |
| Overall Abstention Rate             |   0.3413 |     0.3017 |            0.3025 |             0.4496 |

### Interpretation of RAG-Behavior Results

DPO-v1 and DPO-v2-lightclean improve behavior on evidence-sufficient examples. They achieve higher sufficient-supported-answer rates and lower sufficient-over-refusal rates than the baseline.

However, both DPO variants perform worse on evidence-insufficient examples. They reduce correct refusal and increase overconfident answering, suggesting that weak-label DPO makes the model more willing to answer even when the retrieved context is insufficient.

DPO-v1 + SelfCheck has the best RAG behavior accuracy and the lowest bad answer behavior rate. It substantially improves correct refusal under insufficient evidence. However, it also increases over-refusal on evidence-sufficient examples, showing a clear safety–coverage trade-off.

### Standard Automatic Metrics

| Metric                     | Baseline |     DPO-v1 | DPO-v2-lightclean | DPO-v1 + SelfCheck |
| -------------------------- | -------: | ---------: | ----------------: | -----------------: |
| Diagnostic Overall EM      |   0.2582 | **0.2740** |            0.2684 |             0.2380 |
| Diagnostic Overall F1      |   0.3322 | **0.3492** |            0.3447 |             0.2982 |
| Sufficient EM              |   0.4483 | **0.4685** |            0.4655 |             0.4167 |
| Sufficient F1              |   0.5368 | **0.5552** |            0.5543 |             0.4893 |
| Sufficient Citation Rate   |   0.5374 |     0.9552 |            0.7281 |             0.7315 |
| Insufficient Citation Rate |   0.2468 |     0.7766 |            0.4025 |             0.3354 |
| Overall Citation Rate      |   0.4102 |     0.8770 |            0.5856 |             0.5582 |
| Overall Abstention Rate    |   0.3413 |     0.3017 |            0.3025 |             0.4496 |

### Notes on Automatic Metrics

The automatic evaluation is useful for measuring EM/F1, abstention behavior, and citation-format behavior. However, it has two important limitations:

1. `valid_citation_rate` only checks whether citation IDs refer to retrieved passages. It does not check whether the cited passages actually support the answer.
2. The supported/unsupported split is based on gold-answer string matching, not human-verified evidence support.

Therefore, overall EM/F1 should be interpreted as diagnostic metrics, not primary RAG-faithfulness metrics. In evidence-insufficient examples, matching the gold answer may reflect parametric knowledge rather than grounded answering.

## Inference-Time SelfCheck

The SelfCheck variant applies an output-time verifier to DPO-v1 generations.

The verifier checks whether the generated answer is supported by the retrieved passages. If the generated answer is judged unsupported, the system replaces it with:

```text
I don't know based on the provided evidence.
```

This method does not use gold answers at inference time. It only uses:

```text
question
retrieved passages
generated answer
```

SelfCheck is intended as a safety guardrail rather than a new trained model.

### SelfCheck Finding

SelfCheck substantially reduces hallucination-style bad answer behavior:

```text
DPO-v1 bad_answer_behavior_rate: 0.3898
DPO-v1 + SelfCheck bad_answer_behavior_rate: 0.2767
```

It also improves refusal when evidence is insufficient:

```text
DPO-v1 insufficient_correct_refusal_rate: 0.5057
DPO-v1 + SelfCheck insufficient_correct_refusal_rate: 0.6715
```

However, it increases over-refusal when evidence is sufficient:

```text
DPO-v1 sufficient_over_refusal_rate: 0.1429
DPO-v1 + SelfCheck sufficient_over_refusal_rate: 0.2768
```

This shows that inference-time verification improves safety, but at the cost of answer coverage.

## LLM Verifier Audit

To evaluate grounding more directly, this project also runs a full-validation LLM verifier audit.

The verifier judges each system output using the question, gold answers, retrieved passages, and generated answer. It labels:

* answer type
* gold correctness
* passage support
* citation faithfulness
* hallucination

The verifier is instructed to judge passage support only from the retrieved passages and not to use outside knowledge for evidence support.

### LLM Verifier Results

| Metric                      | Baseline | DPO-v1 | DPO-v2-lightclean |
| --------------------------- | -------: | -----: | ----------------: |
| Answer Rate                 |   65.82% | 69.83% |            69.72% |
| Abstention Rate             |   34.13% | 30.17% |            30.25% |
| Gold Correct / Partial      |   42.27% | 44.24% |            43.79% |
| Passage Supported / Partial |   51.91% | 52.44% |            53.13% |
| Unfaithful Citation Rate    |    9.45% | 37.70% |            17.29% |
| No Citation Rate            |   58.64% | 12.24% |            41.41% |
| Hallucination / Partial     |   16.68% | 19.70% |            19.61% |
| Non-Hallucination Rate      |   82.74% | 80.19% |            80.22% |

### LLM Verifier Findings

The LLM verifier audit shows that DPO-v1 improves answer-seeking behavior, but introduces substantial citation misuse.

DPO-v1 raises gold-correct-or-partial rate from 42.27% to 44.24%, but also increases unfaithful citation rate from 9.45% to 37.70%.

DPO-v2-lightclean reduces the DPO-v1 unfaithful citation rate from 37.70% to 17.29%, a relative reduction of more than 50%. It also preserves most of the answer-quality improvement from DPO-v1.

However, DPO-v2-lightclean does not outperform the baseline on overall hallucination-related metrics. The verifier-labeled hallucination-or-partial rate remains higher than the baseline.

## Main Findings

### 1. Weak-label DPO improves answerability but not necessarily faithfulness

DPO-v1 and DPO-v2-lightclean improve sufficient-evidence answer accuracy and reduce over-refusal. However, they also make the model more willing to answer when retrieved evidence is insufficient.

### 2. Naive DPO amplifies unfaithful citation behavior

DPO-v1 dramatically increases citation frequency. The LLM verifier shows that this citation increase does not correspond to citation faithfulness.

### 3. Cleaning DPO data mitigates citation misuse

DPO-v2-lightclean substantially reduces unfaithful citation compared with DPO-v1 while preserving most of the answer-quality improvement.

### 4. SelfCheck reduces hallucination-style behavior but increases over-refusal

Inference-time SelfCheck provides the strongest reduction in bad answer behavior and overconfident answering, but it also rejects many answerable cases.

### 5. Preference data quality is more important than simply applying DPO

The results suggest that DPO is not automatically hallucination-reducing in RAG. Weak preference labels can encourage answer-seeking and citation-seeking behavior unless the preference data accurately reflects evidence support.

## Takeaway

Weak-label DPO improves answerability only modestly and can amplify citation-style hallucination.

Cleaner DPO construction reduces citation misuse, but does not fully solve answer-level hallucination.

Inference-time SelfCheck is effective as a safety guardrail, reducing unsupported and overconfident answers, but it introduces a safety–coverage trade-off through increased over-refusal.

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

### Evaluate DPO-v1 + SelfCheck

```bash
python scripts/05_eval_generation.py \
  --generation_file data/generation/dpo_v1_selfcheck_val_outputs_top10_full.jsonl \
  --output_file outputs/dpo/dpo_v1_selfcheck_val_metrics_top10_full.json \
  --per_example_output outputs/dpo/dpo_v1_selfcheck_val_eval_top10_full.jsonl
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
    dpo_v1_selfcheck_val_metrics_top10_full.json
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

* The preference data is constructed from weak automatic labels, not human annotations.
* The automatic supported/unsupported split is based on gold-answer string matching.
* Automatic citation validity only checks citation IDs, not citation faithfulness.
* The LLM verifier audit is stronger than string matching, but still depends on the judge model.
* The SelfCheck verifier can be overly conservative and may reject answerable examples.
* The generator is a 7B instruct model, which may limit long-context citation and abstention behavior.
* The project does not include human-labeled preference data.
* The experiments are currently limited to NQ-Open and Wiki DPR passages.

## Future Work

* Build verifier-labeled preference data instead of weak string-match labels.
* Use an LLM or NLI model to check whether passages actually support the chosen answer.
* Train a separate verifier for citation faithfulness and answer support.
* Explore calibrated SelfCheck methods that distinguish unsupported answers from uncertain cases.
* Run prompt ablations with stricter citation rules:

  * If the answer is insufficient evidence, do not cite any passage.
* Evaluate citation faithfulness beyond citation format validity.
* Add list-aware evaluation for multi-answer NQ examples.
* Compare DPO-only, SFT-only, and SFT+DPO variants.
* Test stronger base models with better RAG and citation-following ability.
* Evaluate on multi-hop QA and long-context RAG benchmarks.
