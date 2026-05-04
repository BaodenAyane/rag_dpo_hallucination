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
- [ ] DPO training
- [ ] DPO model generation
- [ ] DPO evaluation
- [ ] Baseline vs DPO comparison

## Current Experimental Setup

- Dataset: NQ-Open
- Corpus: Wiki DPR passages
- Retriever: Pyserini/Lucene BM25
- Generator: Qwen/Qwen2.5-7B-Instruct
- Retrieval top-k for generation: 10
- Generation max new tokens: 64
- Max input length: 4096
- DPO method: LoRA DPO
- DPO data source: baseline RAG generation outputs

## Current DPO Data

DPO data is constructed conservatively to reduce noisy preference pairs.

Current train DPO data statistics:

```json
{
  "final_dpo_examples": 11174,
  "final_supported_pairs": 5587,
  "final_unsupported_pairs": 5587
}