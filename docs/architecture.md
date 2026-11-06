# Architecture and validation boundaries

The report-generation pipeline wraps MedGemma and optionally adds retrieved report
context from BiomedCLIP or ColPali. The QA pipeline retrieves context from an image
and question, then calls MedGemma. `clip-retrieval` is an alternate retrieval-only
path; it copies an indexed report and must never be labeled generated medical evidence.

Core modules:

| Component | Code | Boundary |
|---|---|---|
| Image/report inputs | `src/evaluation/validation.py` | Reject missing images, labels, splits; detect study/patient overlap |
| Report generation | `src/pipelines/report_generation.py` | No missing-index zero-shot fallback or missing-checkpoint relabeling |
| QA | `src/pipelines/rag_qa.py` | Retrieval context is not proof of answer correctness |
| Index construction | `scripts/build_index.py` | Training-only provenance; synthetic indexes kept separately |
| Metrics | `src/evaluation/metrics.py` | Unavailable is null; heuristic is not RadGraph |
| Orchestration | `scripts/run_comparison.py` | Explicit mock/real outputs; no templated scientific findings |

The public repository does not contain a validated MIMIC-CXR index or verified
fine-tuned checkpoint. Real model loading, memory use and inference quality remain
hardware-dependent and unverified by the CPU software audit. Index files are local,
trusted artifacts; pickle-based indexes must not be loaded from untrusted sources.
