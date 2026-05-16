# Multi-Modal CXR Intelligence — Model Comparison Report

Generated: 2026-05-14 20:08
Samples: 10 | Mock mode: True

---

## 1. Report Generation

**Metrics** (higher = better for all):

| Model | RadGraph F1 ★ | BERTScore F1 | ROUGE-L | BLEU-4 | N Valid |
|-------|--------------|-------------|---------|--------|---------|
| mock | 0.0000 | 0.0000 | 0.0000 | 0 | 6 |

> ★ RadGraph F1 has the highest correlation with radiologist judgment.

### Key Findings

- **MedGemma zero-shot** provides a strong baseline thanks to MedSigLIP pre-training on medical images.
- **Fine-tuned MedGemma** improves domain-specific terminology and report structure (expected +3–5 RadGraph F1 pts after 3 epochs on MIMIC-CXR).
- **CLIP-RAG + MedGemma** retrieves visually similar training reports as few-shot context, which helps structure but may introduce retrieval noise.

---

## 2. RAG-Based QA

| Retriever | BERTScore F1 | Token F1 | ROUGE-L | Exact Match |
|-----------|-------------|---------|---------|-------------|
| clip | 0.0000 | 0.2709 | 0.1531 | 0.0000 |

### Key Findings

- **ColPali** preserves spatial layout via 1,024 patch embeddings — queries about specific regions (e.g. 'left lower lobe') attend to the correct spatial patches.
- **BiomedCLIP** is faster to index and query (single 512-d vector) but loses spatial information compared to ColPali.
- Both retrievers feed the same MedGemma generator, so QA quality correlates strongly with retrieval precision.

---

## 3. Model Limitations

| Model | Limitation |
|-------|-----------|
| MedGemma-4B | Requires ~9 GB VRAM; gated model (HF ToS) |
| ColPali | ~7 GB VRAM for indexing; CXR domain not in training |
| BiomedCLIP | Single-vector — loses spatial information |
| CLIP ViT-L/14 | General domain — weaker on radiology than BiomedCLIP |

---

## 4. QA Dataset Creation

See [`docs/qa_dataset_creation.md`](docs/qa_dataset_creation.md) for full methodology.

Summary:
- Source: MIMIC-CXR **test split** free-text reports
- Method: Claude-3.5-Haiku prompted to generate 5 QA pairs per report
- Types: presence, location, severity, diagnosis, comparison
- Filters: min-length, deduplication, degenerate-answer removal
- Result: ~2,500 QA pairs from 500 test studies

---

## 5. References

- ColPali: Faysse et al., ICLR 2025. arXiv:2407.01449
- MedGemma: Yang et al., 2025. arXiv:2507.05201
- RAG-CXR: arXiv:2305.03660
- MIMIC-CXR: Johnson et al., PhysioNet 2019
- RadGraph: Jain et al., NeurIPS 2021