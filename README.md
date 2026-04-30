# RAG-DPO Hallucination

This project studies whether preference optimization can reduce hallucination in retrieval-augmented generation.

Current pipeline:
1. Download NQ-Open and Wiki DPR passages
2. Build BM25 retrieval baseline
3. Retrieve top-k passages for NQ-Open questions
4. Generate baseline RAG answers
5. Construct SFT/DPO preference pairs
6. Fine-tune generator with DPO
7. Evaluate factuality and hallucination

## Status

- [x] Data download
- [x] BM25 retrieval baseline
- [ ] Baseline generation
- [ ] Evidence support filtering
- [ ] SFT data construction
- [ ] DPO pair construction
- [ ] SFT training
- [ ] DPO training
- [ ] Evaluation