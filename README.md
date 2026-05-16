#  Multi-Modal Chest X-Ray Intelligence System

A dual-mode medical AI system for chest radiograph interpretation:
- **Mode 1 — Report Generation**: Upload a CXR image → get a structured radiology report
- **Mode 2 — RAG-Based QA**: Ask clinical questions about a CXR, grounded in retrieved evidence

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│            CXR INTELLIGENCE SYSTEM                          │
├─────────────────────────────────────────────────────────────┤
│  Mode 1: Report Generation                                  │
│    CXR Image → MedGemma-4B-it → Findings + Impression      │
│    CXR Image → BiomedCLIP Retrieval + LLM → Report (RAG)   │
│                                                             │
│  Mode 2: RAG-Based QA                                       │
│    (Image + Question) → ColPali Retrieve → MedGemma → Ans  │
│    (Image + Question) → CLIP Retrieve → MedGemma → Answer  │
│                                                             │
│  Knowledge Base: ColPali-indexed MIMIC-CXR training set    │
└─────────────────────────────────────────────────────────────┘
```

## Models Used

| Model | Role | HuggingFace ID |
|-------|------|----------------|
| MedGemma-4B-it | Report generation + QA generation | `google/medgemma-4b-it` |
| ColPali-v1.2 | Multimodal visual retrieval (RAG) | `vidore/colpali-v1.2` |
| BiomedCLIP | Image-text similarity retrieval | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` |
| MedSigLIP | (inside MedGemma) Medical vision encoder | bundled |

## Dataset

**MIMIC-CXR** (Kaggle version):  
[simhadrisadaram/mimic-cxr-dataset](https://www.kaggle.com/datasets/simhadrisadaram/mimic-cxr-dataset)

The `text` column contains free-text radiology reports (Findings + Impression). Images are chest X-rays in JPEG format.

---

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/your-username/cxr-intelligence.git
cd cxr-intelligence
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your tokens:
#   HF_TOKEN       → HuggingFace token (required for MedGemma gated model)
#   KAGGLE_USERNAME / KAGGLE_KEY → For dataset download
#   ANTHROPIC_API_KEY → For QA dataset generation (optional)
```

### 3. Download Dataset

```bash
python scripts/download_data.py --dataset mimic-cxr --sample 10000
```

### 4. Build RAG Index

```bash
python scripts/build_index.py --model colpali --data-dir data/processed --output data/index
```

### 5. Run Demo

```bash
python app/app.py
```

---

## Quickstart (5 images, no GPU needed — mock mode)

```bash
python scripts/quickstart.py --mock
```

---

## Evaluation

```bash
# Report generation evaluation (500 test samples)
python scripts/evaluate_report_gen.py \
    --model medgemma \
    --data-dir data/processed \
    --split test \
    --n-samples 500 \
    --output results/report_gen_medgemma.json

# QA evaluation
python scripts/evaluate_qa.py \
    --qa-dataset data/qa_dataset.json \
    --retriever colpali \
    --output results/qa_colpali.json
```

---

## Fine-tuning MedGemma (QLoRA)

```bash
python scripts/finetune_medgemma.py \
    --data-dir data/processed \
    --output-dir checkpoints/medgemma-cxr \
    --epochs 3 \
    --batch-size 4 \
    --lora-rank 16
```

---

## Project Structure

```
cxr-intelligence/
├── src/
│   ├── data/           # Dataset classes, preprocessing, report parsing
│   ├── models/         # MedGemma, ColPali, CLIP wrappers
│   ├── pipelines/      # Report generation & RAG QA pipelines
│   ├── rag/            # Indexing, retrieval, vector store
│   ├── evaluation/     # Metrics: RadGraph F1, BERTScore, BLEU, QA eval
│   └── qa_dataset/     # Synthetic QA pair generation
├── scripts/            # CLI entrypoints
├── app/                # Gradio demo application
├── configs/            # YAML configuration files
├── tests/              # Pytest unit tests
├── docs/               # Architecture docs, QA dataset methodology
└── results/            # Evaluation outputs (JSON + CSV)
```

---

## QA Dataset Creation

Since MIMIC-CXR has no official QA annotations, we generate synthetic QA pairs from reports using an LLM. See [docs/qa_dataset_creation.md](docs/qa_dataset_creation.md) for full methodology.

```bash
python scripts/generate_qa_dataset.py \
    --reports-csv data/processed/test_reports.csv \
    --output data/qa_dataset.json \
    --n-questions 5 \
    --provider anthropic
```

---

## GPU Requirements

| Mode | VRAM | Notes |
|------|------|-------|
| MedGemma 4B (bfloat16) | ~9 GB | Recommended |
| MedGemma 4B (4-bit) | ~5 GB | With bitsandbytes |
| ColPali v1.2 | ~7 GB | Indexing |
| Mock mode | 0 | CPU testing only |

---

## Citation

```bibtex
@misc{cxr-intelligence,
  title={Multi-Modal Chest X-Ray Intelligence System},
  year={2025},
  note={Dual-mode: Report Generation and RAG-based QA}
}
```

Key papers: ColPali (arXiv:2407.01449), MedGemma (arXiv:2507.05201), 
RAG-CXR (arXiv:2305.03660), MIMIC-CXR (PhysioNet).
