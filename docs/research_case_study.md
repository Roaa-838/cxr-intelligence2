# Reliable Multimodal AI for Chest X-ray Report Generation and QA

**CXR-Intelligence | Technical case study | November 6, 2026**

## Abstract

CXR-Intelligence implements chest X-ray report generation and retrieval-augmented
question answering with MedGemma, BiomedCLIP and ColPali components. This case study
reports an engineering audit that separates mock execution from real-model evidence,
rejects invalid evaluation inputs, and removes unsupported performance conclusions.
Software tests and synthetic smoke execution were verified; clinical accuracy and
comparative real-model performance remain unevaluated in this audit.

## Problem and research questions

Retrieved reports can contain findings unsupported by the query image. The proposed
research asks whether visual retrieval improves report grounding, how different
retrievers compare under controlled conditions, and when negation or laterality
errors occur. These are research questions, not established findings.

## Implemented system and contribution scope

The code provides an image-to-report pathway and an image/question-to-retrieval-to-
answer pathway. BiomedCLIP and ColPali wrappers build visual indexes; MedGemma generates
reports or answers using the image and optional context. Evaluation entry points
coordinate inference and text-similarity scoring. A Gradio interface supports demonstrations.

This audit adds shared input validation, provenance checks, truthful metric naming,
mode-separated outputs, regression tests and evidence-focused documentation. The
repository does not establish who authored each original component. Personal/team
contribution attribution must be confirmed before using this study as a first-person
application statement.

## Data and method

No patient dataset was processed in this audit. The software smoke test uses ten
synthetic image/report fixtures and ten synthetic QA fixtures, with mock models.
These fixtures test execution and serialization, not image understanding.

For future real runs, metadata must provide actual images, reference reports, study
IDs and explicit splits. Patient IDs are required for RAG provenance. The loader
checks study/patient overlap across metadata splits; the evaluator checks that index
entries are training-only, real and disjoint from evaluation patients. Missing images,
labels, requested splits, checkpoints or indexes fail explicitly. The checks rely on
accurate supplied metadata and do not independently establish dataset authorization.

## Verified results

| Check | Observed outcome | Interpretation |
|---|---|---|
| Automated tests | 76 passed; 26 new integrity tests | Engineering verification |
| Mock report execution | 10/10 outputs | Functional smoke test |
| Mock QA execution | 10/10 outputs | Functional smoke test |
| Real-model comparison | Not run | No performance conclusion |
| Clinical validation | Not performed | Not suitable for diagnostic claims |

The mock lexical scores are retained in machine-readable evidence but are not presented
as model accuracy. BERTScore and RadGraph were deliberately omitted from mock runs.
Failed optional metrics use null and a status rather than zero. The separately named
Clinical entity-overlap heuristic F1 does not reproduce RadGraph or validate negation.

## Failure analysis and limitations

The earlier comparison path could supply missing images, create synthetic inputs
without a mock flag, and generate conclusions independently of measured results.
The QA path could replace unavailable images with blank ones. Historical metric
failures could appear as zero, and a keyword heuristic carried the RadGraph label.
Regression tests now exercise these failure conditions.

No evidence here establishes retriever superiority, fine-tuning gains, hallucination
reduction or clinical safety. Legacy retrieval aggregates remain unverified. Patient-
level excerpts and identifiers were removed from the current result files, but original
content remains in Git history. Data authorization, model revision pinning, official
split provenance, annotation quality and clinical review remain unresolved prerequisites.

## Next experiment

Evaluate a permitted patient-disjoint sample with fixed generation settings: MedGemma
zero-shot versus retrieval-augmented generation. Keep cases, prompts and retrieval depth
comparable; document all deviations. Manually review supported/unsupported findings,
negation and laterality using a defined protocol, and distinguish language similarity
from grounding. Without qualified clinical review, call it report-grounding evaluation.

## Reproducibility and references

`python -m pytest -q`

`python scripts/run_comparison.py --mock --n-samples 10`

Evidence: `results/audit_2026_11_06/`; source baseline:
`13374a42fd536c22317d5aca434e19fc74892b5e` plus the audit changes.

Repository: https://github.com/Roaa-838/cxr-intelligence2

Dataset reference: Johnson et al., MIMIC-CXR v2.1.0, PhysioNet (2024),
https://physionet.org/content/mimic-cxr/2.1.0/ . Dataset use is subject to its access terms.
This case study is a technical report, not a peer-reviewed publication.
