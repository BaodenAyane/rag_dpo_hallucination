# Baseline vs DPO-v1 vs DPO-v2-lightclean Results

This document summarizes validation results for three systems:

1. Baseline RAG generator
2. DPO-v1 tuned generator
3. DPO-v2-lightclean tuned generator

## Setup

- Dataset: NQ-Open validation set
- Number of validation examples: 3610
- Corpus: Wiki DPR passages
- Retriever: Pyserini/Lucene BM25
- Generator: Qwen/Qwen2.5-7B-Instruct
- Retrieval top-k for generation: 10
- Generation max new tokens: 64
- Evaluation: EM, F1, citation rate, valid citation rate, abstention rate, and retrieved-answer support split

## Methods

### Baseline

The baseline model is Qwen2.5-7B-Instruct prompted to answer using retrieved BM25 passages.

### DPO-v1

DPO-v1 uses weak preference pairs constructed from baseline behavior:

- Supported context + baseline abstention:
  - chosen = gold answer with citation
  - rejected = insufficient evidence
- Unsupported context + baseline non-abstention:
  - chosen = insufficient evidence
  - rejected = baseline answer

DPO-v1 used a 1:1 supported/unsupported ratio.

### DPO-v2-lightclean

DPO-v2-lightclean modifies DPO-v1 in three ways:

1. Uses an unsupported-heavy final ratio of roughly 1:2.
2. Adds an explicit no-citation instruction for unsupported chosen responses.
3. Filters unsupported pairs where the baseline short answer appears to be directly supported by retrieved passages.

This version is designed to reduce the unsupported citation explosion observed in DPO-v1.

## Validation Results

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

## Deltas

### DPO-v1 vs Baseline

| Metric | Delta |
|---|---:|
| Exact Match | +0.0158 |
| F1 | +0.0170 |
| Citation Rate | +0.4668 |
| Supported F1 | +0.0183 |
| Supported Abstention Rate | -0.0281 |
| Unsupported Citation Rate | +0.5297 |
| Unsupported Abstention Rate | -0.0544 |

### DPO-v2-lightclean vs Baseline

| Metric | Delta |
|---|---:|
| Exact Match | +0.0102 |
| F1 | +0.0125 |
| Citation Rate | +0.1753 |
| Supported F1 | +0.0175 |
| Supported Abstention Rate | -0.0335 |
| Unsupported Citation Rate | +0.1557 |
| Unsupported Abstention Rate | -0.0456 |

### DPO-v2-lightclean vs DPO-v1

| Metric | Delta |
|---|---:|
| Exact Match | -0.0055 |
| F1 | -0.0045 |
| Citation Rate | -0.2914 |
| Supported F1 | -0.0009 |
| Supported Abstention Rate | -0.0054 |
| Unsupported Citation Rate | -0.3741 |
| Unsupported Abstention Rate | +0.0089 |

## Main Findings

DPO-v1 improves answer accuracy and citation formatting, but it also sharply increases citation behavior on unsupported examples.

DPO-v2-lightclean preserves most of the supported-answer gains from DPO-v1 while substantially reducing unsupported citation rate.

The main tradeoff is:

- DPO-v1 has the best EM/F1.
- DPO-v2-lightclean has much lower unsupported citation rate than DPO-v1.
- Neither DPO-v1 nor DPO-v2-lightclean restores unsupported abstention rate to the baseline level.

## Interpretation

DPO-v1 appears to make the model more answer-seeking and citation-seeking. This improves supported-answer performance, but it also encourages citation-formatted answers in unsupported contexts.

DPO-v2-lightclean partially fixes this by adding cleaner unsupported pairs and explicit no-citation chosen responses. It reduces unsupported citation rate from 0.7766 to 0.4025 while keeping supported F1 nearly unchanged.

However, unsupported abstention remains below the baseline. This suggests that weak-label DPO alone is not sufficient to fully teach robust evidence-grounded abstention.

## Takeaway

DPO can improve RAG answerability and citation behavior, but naive weak-label DPO may amplify citation-style hallucination.

A cleaner DPO construction reduces this effect, but further work is needed to improve abstention behavior on unsupported retrieval contexts.

## Future Directions

- Add stricter prompt rules:
  - If the answer is insufficient evidence, do not cite any passage.
- Run prompt ablation without retraining.
- Add verifier-based support filtering.
- Use NLI or LLM-based citation faithfulness checks.
- Compare DPO-only, SFT-only, and SFT+DPO variants.
- Evaluate citation faithfulness beyond citation format validity.
