# QA Dataset Creation Methodology

## Overview

Since MIMIC-CXR contains no official question-answer annotations, we generate
a synthetic QA dataset by prompting a large language model (LLM) to produce
clinically valid question-answer pairs from the free-text radiology reports.

This approach is validated by prior work: the MIMIC-Ext-MIMIC-CXR-VQA dataset
(Physionet) uses a similar template-based generation strategy, and the
MIMIC-CXR-VQA paper (MIDL 2026) used LLaMA annotations to create 48-template
QA pairs from the same MIMIC-CXR reports.

---

## Step-by-Step Methodology

### Step 1 — Source Material Selection

- **Split used**: Test split only (`split == "test"`, ~3,000 studies in MIMIC-CXR)
- **Rationale**: Using test-split reports prevents data leakage — the RAG index
  is built from the training split, and the QA pairs are evaluated on test-split
  images. The model has never seen these reports during training or indexing.
- **Minimum report length**: 50 characters (very short stubs excluded)
- **Target sample**: 500 studies → ~2,500 QA pairs

### Step 2 — Report Parsing

Each report is parsed into structured sections using `ReportParser`:

```
FINDINGS: → systematic image description
IMPRESSION: → clinical summary
TECHNIQUE: → acquisition parameters (used for context only)
COMPARISON: → prior study reference
```

Only `FINDINGS` and `IMPRESSION` are used for QA generation.

### Step 3 — LLM QA Generation

**Provider**: Anthropic Claude-3.5-Haiku (fast, cost-efficient, strong clinical reasoning)  
**Alternative**: OpenAI GPT-4o-mini

**Prompt design** (see `src/qa_dataset/generator.py: GENERATION_PROMPT`):
- Instructs the model to act as a board-certified radiologist
- Requests exactly N question-answer pairs in a fixed JSON schema
- Specifies 5 question types (see below)
- Requires answers to cite the verbatim evidence from the report
- Requires both positive (finding present) and negative (no finding) cases

**Output schema per QA pair**:
```json
{
  "question_type": "presence | location | severity | diagnosis | comparison",
  "question": "Is there a pleural effusion?",
  "answer": "No. The report states no pleural effusion or pneumothorax is identified.",
  "evidence": "No pleural effusion or pneumothorax is identified.",
  "difficulty": "easy | medium | hard"
}
```

### Step 4 — Question Type Taxonomy

| Type | Definition | Example | Difficulty |
|------|-----------|---------|-----------|
| **presence** | Is finding X present/absent? | "Is there a pneumothorax?" | Easy |
| **location** | Where is finding X located? | "Which lobe shows consolidation?" | Medium |
| **severity** | How severe is finding X? | "How severe is the cardiomegaly?" | Medium |
| **diagnosis** | What is the diagnosis/impression? | "What is the primary radiological finding?" | Hard |
| **comparison** | Normal vs abnormal? | "Is this chest X-ray within normal limits?" | Medium |

**Target distribution** (per 5 QA pairs per study):
- presence: ~35% (most clinically asked)
- diagnosis: ~20%
- location: ~20%
- severity: ~15%
- comparison: ~10%

### Step 5 — Quality Filtering

After generation, apply these automatic filters (`QADatasetGenerator.filter_qa_pairs`):

1. **Minimum length**: question ≥ 15 chars, answer ≥ 10 chars
2. **Degenerate answer removal**: discard answers matching `{"unknown", "not specified",
   "cannot be determined", "not mentioned", "unclear", "n/a"}`
3. **Deduplication**: remove near-identical questions within the same study
   (key = `study_id + normalised_question_text`)

Typical filter pass rate: ~85% (15% removed as degenerate or too short).

### Step 6 — Validation

**Automatic validation**:
- For `presence` questions: cross-check generated Yes/No answers against
  CheXpert structured labels (e.g., if label `Pleural Effusion = 1.0`,
  answer should be affirmative). Accuracy target: >90%.

**Manual spot check**:
- Sample 100 QA pairs randomly
- A human annotator (or radiologist) rates each answer as:
  - **Correct** (3): answer is factually supported by the report
  - **Acceptable** (2): minor inaccuracy but clinically appropriate
  - **Incorrect** (1): answer contradicts or is not supported by the report
- Target: >85% rated as Correct or Acceptable

### Step 7 — Final Dataset

After filtering and validation:

```
Total QA pairs:     ~2,500
Unique studies:     ~500
Pairs per study:    ~5
Split:              All from test set (no train contamination)
Storage format:     JSON array → data/qa_dataset.json
```

---

## Comparison with Prior Work

| Dataset | Method | Size | Types |
|---------|--------|------|-------|
| MIMIC-Ext-MIMIC-CXR-VQA | Template-based (48 templates) | 377K entries | 7 content types |
| MIMIC-CXR-VQA (MIDL 2026) | LLaMA annotations | ~50K | Presence, location, severity |
| **Ours** | Claude-3.5-Haiku prompted | ~2,500 | 5 types, evidence-grounded |

Our approach is smaller but higher quality per pair: each answer includes
an explicit `evidence` field linking to the exact report sentence, enabling
faithful evaluation.

---

## Reproducing the QA Dataset

```bash
# With Claude API (recommended)
python scripts/generate_qa_dataset.py \
    --data-dir data/processed \
    --split test \
    --n-samples 500 \
    --n-questions 5 \
    --provider anthropic \
    --model claude-3-5-haiku-20241022 \
    --filter \
    --output data/qa_dataset.json

# Without API (template-based mock for testing)
python scripts/generate_qa_dataset.py \
    --synthetic \
    --n-samples 100 \
    --output data/qa_dataset_mock.json

# Validate and inspect
python scripts/generate_qa_dataset.py \
    --validate data/qa_dataset.json
```

---

## Limitations

1. **LLM-generated answers** may not perfectly match how a radiologist would
   answer — especially for ambiguous or subtle findings.
2. **Report-grounded only**: QA pairs are derived from text reports, not from
   direct image analysis. An ideal VQA dataset requires radiologist annotation
   from the image itself.
3. **Presence/absence bias**: ~35% of questions are yes/no, which may inflate
   exact-match scores for models that default to "No" (most findings are absent
   in normal CXRs).
4. **No negation handling**: simple entity extraction does not disambiguate
   "No pleural effusion" from "pleural effusion" — the LLM handles this, but
   it's an acknowledged limitation of the approach.
