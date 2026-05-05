# Baseline vs DPO-v1 Results

This document summarizes the validation results of the baseline RAG generator and the DPO-v1 tuned generator.

## Setup

- Dataset: NQ-Open validation set
- Number of validation examples: 3610
- Corpus: Wiki DPR passages
- Retriever: Pyserini/Lucene BM25
- Generator: Qwen/Qwen2.5-7B-Instruct
- Retrieval top-k for generation: 10
- Generation max new tokens: 64
- DPO method: LoRA DPO

## Baseline vs DPO-v1

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
| Supported Valid Citation Rate | 0.5374 | 0.9552 | +0.4177 |
| Supported Abstention Rate | 0.1709 | 0.1429 | -0.0281 |
| Unsupported EM | 0.0139 | 0.0241 | +0.0101 |
| Unsupported F1 | 0.0694 | 0.0847 | +0.0153 |
| Unsupported Citation Rate | 0.2468 | 0.7766 | +0.5297 |
| Unsupported Valid Citation Rate | 0.2468 | 0.7766 | +0.5297 |
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

## Interpretation

The current support label is weak:

```text
answer_in_retrieved = whether any gold answer string appears in retrieved passages
```

This label does not guarantee that the retrieved passages actually support the question-answer relation. As a result, some preference pairs may encourage the model to trust retrieved passages too aggressively.

DPO-v1 should therefore be interpreted as an answerability-oriented DPO baseline rather than a fully hallucination-reducing method.

## Takeaway

DPO-v1 shows that preference optimization can improve RAG answerability and citation formatting, but weak-label DPO alone is not sufficient to reduce unsupported citation-style hallucination.

Future work should improve the support signal with verifier-based filtering or citation faithfulness evaluation.