# Model comparison: protocol and evidence status

The previous expected-number tables and qualitative examples were not supported
by traceable experiments. They have been withdrawn rather than cited as results.

| Configuration | Current status | Proposed comparison |
|---|---|---|
| MedGemma zero-shot | Implemented; controlled real evaluation pending | Fixed baseline |
| Fine-tuned MedGemma | Training script exists; checkpoint/results unverified | Only if a verified checkpoint is available |
| BiomedCLIP-RAG + MedGemma | Implemented; controlled evaluation pending | Same generator, prompt and cases |
| ColPali + MedGemma | Implemented; controlled evaluation pending | Same generator and QA cases |

## Metric interpretation

- BLEU and ROUGE measure lexical similarity, not diagnostic accuracy.
- BERTScore measures embedding similarity; unavailable weights/dependencies are reported as null.
- RadGraph F1 is reported only if the official evaluator executes successfully.
- Clinical entity-overlap heuristic F1 is separately named and cannot validate negation or laterality.
- QA token F1 uses unique-token sets; exact match uses normalized answer text.
- Error or empty inference outputs invalidate the aggregate run rather than being scored as predictions.

## Planned experiment

Use a permitted held-out sample with patient-disjoint train/evaluation partitions.
Keep generation settings, input cases and retrieval depth fixed across methods.
Record model revisions, effective precision/device, environment, input counts,
evaluator versions and failures. Model IDs alone are not immutable revisions;
pin downloaded revisions before a reproducibility claim about real inference.

Define retrieval relevance independently: retrieving the held-out query's own report
is not a valid training-only RAG target. Annotate suitable training evidence or
report retrieval quality separately from answer quality.

Compare unsupported findings, negation and laterality errors under a documented
review protocol. Without qualified clinical review, describe this as report-grounding
evaluation. No method is assumed to outperform another.

See [audit](research_credibility_audit.md) and [current results status](../results/comparison_report.md).
