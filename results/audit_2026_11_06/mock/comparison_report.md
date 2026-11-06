# CXR evaluation report

Evaluation mode: **mock**.
Requested maximum samples per task: 10.

Mock runs test software only; they do not measure medical model performance.

| Task | Method | Valid / input | Metric | Value |
|---|---|---|---|---|
| Report generation | mock | 10 / 10 | RadGraph F1 | Unavailable |
| Report generation | mock | 10 / 10 | BERTScore F1 | Unavailable |
| Report generation | mock | 10 / 10 | ROUGE-L | 0.0952 |
| Report generation | mock | 10 / 10 | BLEU-4 | 0.0063 |
| QA | mock | 10 / 10 | BERTScore F1 | Unavailable |
| QA | mock | 10 / 10 | ROUGE-L | 0.2143 |
| QA | mock | 10 / 10 | Token F1 (unique tokens) | 0.2727 |
| QA | mock | 10 / 10 | Exact match | 0.0000 |

## Interpretation

Unavailable metrics are null, not zero. The entity-overlap heuristic is not RadGraph and does not validate negation or laterality.
No superiority, fine-tuning benefit, hallucination reduction, or clinical accuracy is inferred automatically.
See the per-method JSON files for evaluator status, versions, failures, and effective model settings.
