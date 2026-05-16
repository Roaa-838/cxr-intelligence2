# System Architecture

## Overview

The CXR Intelligence System is a dual-mode multi-modal AI pipeline for
chest X-ray interpretation. Both modes share a common vision-language
model (MedGemma-4B-it) but differ in how context is assembled:

```
┌────────────────────────────────────────────────────────────────────┐
│                  CXR INTELLIGENCE SYSTEM                           │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  MODE 1 — REPORT GENERATION                                  │  │
│  │                                                              │  │
│  │  CXR Image ──► [MedSigLIP encoder] ──► [Gemma-3 4B LLM]    │  │
│  │                                              │               │  │
│  │         Optionally: BiomedCLIP retrieve      │               │  │
│  │         top-K reports as few-shot context ───┘               │  │
│  │                                              ▼               │  │
│  │                                    FINDINGS + IMPRESSION      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  MODE 2 — RAG-BASED QA                                       │  │
│  │                                                              │  │
│  │  CXR Image + Question                                        │  │
│  │       │                                                      │  │
│  │       ├──► [ColPali: 1024 patches × 128d] ──► MaxSim ──┐    │  │
│  │       │    OR                                           │    │  │
│  │       └──► [BiomedCLIP: 512d vector]  ──► CosSim ──────┘    │  │
│  │                                              ▼               │  │
│  │                                    Top-K similar studies     │  │
│  │                                         (reports)            │  │
│  │                                              │               │  │
│  │       ┌──────────────────────────────────────┘               │  │
│  │       ▼                                                      │  │
│  │  [MedGemma-4B-it: image + question + retrieved context]      │  │
│  │       ▼                                                      │  │
│  │  Grounded clinical answer                                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Knowledge Base (indexed offline):                                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  10K training CXRs → ColPali embeddings → numpy array       │  │
│  │  10K training CXRs → BiomedCLIP embeddings → numpy array    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### MedGemma-4B-it (Primary VLM)

- **Role**: Report generation (Mode 1) + answer generation (Mode 2)
- **Architecture**: Gemma 3 4B language model + MedSigLIP vision encoder
- **Vision encoder**: MedSigLIP — SigLIP fine-tuned on medical image-text pairs
  (chest X-rays, pathology slides, ophthalmology images, dermatology)
- **Input resolution**: 448×448 pixels (or 896×896 for higher detail)
- **Context window**: 128K tokens
- **Fine-tuning**: QLoRA (rank 16, NF4 4-bit) on MIMIC-CXR train split
- **Inference**: `device_map="auto"` with optional bitsandbytes 4-bit quantization

### ColPali-v1.2 (Primary Retriever)

- **Role**: Multimodal visual retrieval for RAG Mode 2
- **Architecture**: PaliGemma-3B (SigLIP-So400m + Gemma 2B)
- **Embedding**: 1,024 patch vectors × 128 dimensions per image (32×32 grid)
- **Matching**: Late interaction MaxSim (ColBERT-style):
  `score(q,d) = Σ_i max_j q_i · d_j`
- **Key advantage**: Spatial grounding — patches correspond to image regions,
  so queries about specific anatomical locations attend to correct patches
- **Index format**: numpy float16 array (N × 1024 × 128)

### BiomedCLIP (Comparison Retriever)

- **Role**: Single-vector image-text retrieval (comparison to ColPali)
- **Architecture**: ViT-B/16 + PubMedBERT, trained on 15M biomedical pairs
- **Embedding**: 512-dimensional L2-normalised vector
- **Matching**: Cosine similarity (dot product after normalisation)
- **Advantage**: 10× faster than ColPali; fits in 2 GB GPU
- **Limitation**: Loses spatial information; single vector for entire image

### Vector Store (Qdrant / FAISS)

- **Qdrant in-memory**: Used for development and small-scale testing
- **Qdrant on-disk**: Recommended for production with >50K studies
- **FAISS**: Fallback when Qdrant is unavailable; IndexFlatIP (exact search)

---

## Data Flow

### Offline Indexing (one-time)

```
MIMIC-CXR train set (10K studies)
         │
         ├──► CXRPreprocessor (448×448 RGB, CLAHE)
         │           │
         │     ┌─────┴──────┐
         │     ▼            ▼
         │  ColPali      BiomedCLIP
         │  embed        embed
         │     │            │
         │  (N,1024,128)  (N,512)
         │     │            │
         │  colpali_index.npy   clip_index.pkl
         └──────────────────────────────────────
```

### Online Inference — Mode 1

```
CXR Image (user upload)
     │
     ▼
CXRPreprocessor → PIL 448×448
     │
     ▼
[Optional] BiomedCLIP embed → retrieve top-K reports → context string
     │
     ▼
MedGemmaModel._run_inference(image, prompt + context)
     │
     ▼
raw text → ReportParser → {findings, impression}
```

### Online Inference — Mode 2

```
CXR Image + Question (user input)
     │
     ├──► ColPaliRetriever.embed_text(question) → (N_q, 128)
     │    ColPaliRetriever.score_query_against_index()
     │    → MaxSim scores → top-K metadata
     │
     ├──► RAGQAPipeline._build_context(retrieved)
     │    → formatted context string
     │
     └──► MedGemmaModel.answer_question(image, question, context)
          → answer string
```

---

## File Structure

```
src/
├── config.py                  ← YAML + env var config loader
├── data/
│   ├── dataset.py             ← MIMICCXRDataset + SyntheticCXRDataset
│   ├── preprocessing.py       ← CXRPreprocessor (CLAHE, resize, normalise)
│   └── report_parser.py       ← Section splitter + entity extractor
├── models/
│   ├── base.py                ← Abstract BaseVLM, BaseRetriever, mocks
│   ├── prompts.py             ← All prompt templates (report gen + QA)
│   ├── medgemma.py            ← MedGemma wrapper + QLoRA fine-tuning
│   ├── clip_retriever.py      ← CLIP / BiomedCLIP embed + retrieve
│   └── colpali_retriever.py   ← ColPali patch embed + MaxSim retrieve
├── pipelines/
│   ├── report_generation.py   ← Mode 1 pipeline (zero-shot + CLIP-RAG)
│   └── rag_qa.py              ← Mode 2 pipeline (ColPali/CLIP + MedGemma)
├── rag/
│   ├── indexer.py             ← High-level index orchestrator
│   └── store.py               ← Qdrant / FAISS vector store
├── evaluation/
│   ├── metrics.py             ← BLEU, ROUGE, BERTScore, RadGraph F1
│   └── qa_evaluator.py        ← QA metrics + LLM-as-judge
└── qa_dataset/
    └── generator.py           ← LLM-based QA pair generation
```

---

## Design Decisions

### Why MedGemma over GPT-4V?

1. **Open weights**: Reproducible research; no API dependency or cost
2. **Medical pre-training**: MedSigLIP encoder specifically tuned on CXR data
3. **Proven on MIMIC-CXR**: RadGraph F1 = 30.3 after fine-tuning (paper result)
4. **Runnable on single GPU**: A100 40GB or T4 16GB (4-bit)

### Why ColPali over Dense CLIP for RAG?

1. **Spatial grounding**: 32×32 patch grid → regional attention
2. **No OCR required**: Processes the image directly as a visual document
3. **Late interaction**: Richer matching than single cosine similarity
4. **Higher retrieval accuracy** on visual documents (ViDoRe benchmark SOTA)

### Why CLIP/BiomedCLIP as comparison?

1. Medical domain pre-training (BiomedCLIP on 15M PMC image-text pairs)
2. Much faster indexing and retrieval than ColPali
3. Provides a strong quantitative baseline to show ColPali's added value
4. Standard approach in prior CXR RAG literature (arXiv:2305.03660)
