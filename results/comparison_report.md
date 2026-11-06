# Evaluation status - November 6, 2026

The former report described a **mock run**, not a MedGemma model comparison.
Its unmeasured findings have been withdrawn. Historical aggregate artifacts
are retained under [legacy](legacy/README.md), separately from new mock runs.

| Method | Evidence | Research conclusion |
|---|---|---|
| Mock report/QA pipelines | Software smoke test only | No model-performance conclusion |
| MedGemma zero-shot | No verified real evaluation in this audit | Pending |
| BiomedCLIP-RAG + MedGemma | No verified controlled real evaluation | Pending |
| ColPali + MedGemma | No verified controlled real evaluation | Pending |

The empty model objects in the old evaluation summary do not establish scores
or explain the cause of failure. The heuristic metric is named **Clinical
entity-overlap heuristic F1**, not RadGraph F1.

See [audit](../docs/research_credibility_audit.md) and
[case study](../docs/research_case_study.md) for scope and limitations.
