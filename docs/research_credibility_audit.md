# CXR-Intelligence: Research-Credibility Audit

Audit date: **November 6, 2026**. Baseline commit:
`13374a42fd536c22317d5aca434e19fc74892b5e`.

## Outcome

The evaluation code and public documentation have been corrected to distinguish
software testing from model-performance evidence. **Real-model research evaluation
remains pending.** This is a source-code and software-validation audit, not a clinical
validation, verification of dataset access, or certification of every application path.

## Findings and repairs

| Finding | Repair | Remaining boundary |
|---|---|---|
| CSV comparison used `image=None` | Shared strict image/report loader | Requires actual authorized images |
| Missing files could invoke synthetic data | Synthetic inputs only under explicit mock mode | Other training/demo utilities retain their own workflows |
| QA used blank images and missing answers | Validate all selected inputs before inference | Semantic correctness of annotations needs review |
| Validation split could replace test | Require requested split; no fallback | Official split provenance is still the user's responsibility |
| Unverified indexes could mix patients | Preserve patient/study/split/data/model provenance; reject overlaps and legacy indexes | Supplied metadata must be truthful |
| Missing RAG index silently became zero-shot | Fail explicitly | Rebuild indexes locally |
| Missing fine-tuned checkpoint used base model | Fail explicitly | Fine-tuned loader still needs real checkpoint verification |
| Environment could activate mock weights in a real run | Reject conflicting mock setting | Use explicit `--mock` for tests |
| Heuristic called RadGraph | Separate `clinical_entity_overlap_f1` | Heuristic does not score clinical correctness |
| Failed metric looked like zero | Null plus evaluator error/status | Optional evaluators must be installed for real scoring |
| Failed inference entered metric aggregation | Reject incomplete/error/empty batches; nonzero CLI exit | Failure details remain local |
| Reports generated unmeasured findings | Remove static claims and expected scores | Scientific conclusions require actual experiments |
| Original results mixed modes | Archive legacy aggregates; new outputs under mock/real directories | Legacy provenance cannot be reconstructed |
| Published patient-level examples | Remove identifiers, report excerpts and local paths from current result artifacts | Originals remain in Git history |
| Install backend was invalid | Use `setuptools.build_meta` | Full GPU environment not reproduced |

The legacy `medgemma` and `clip-rag` empty summary objects contain no measurements.
Their failure causes were not recorded; this audit does not invent them. Historical
retrieval-only records are classified **unverified**, not automatically mock. Old
plots were withdrawn because their provenance was insufficient.

## Verification

- Full test suite: **76 passed**, including 26 new research-integrity regression tests.
- Mock comparison: **10 report samples and 10 QA samples completed**, CPU, synthetic
  fixtures, no real model weights. Learned BERTScore and RadGraph deliberately not run.
- Invalid real-data inputs: rejected by regression tests; no blank/synthetic substitute.
- Python compilation and whitespace/diff checks passed.
- Saved evidence: [test run](../results/audit_2026_11_06/verification.txt),
  [mock report](../results/audit_2026_11_06/mock/comparison_report.md),
  [metrics/runtime](../results/audit_2026_11_06/mock/all_metrics.json).

The recorded source commit is the baseline plus a dirty working tree because checks
ran before committing the fixes. Use this branch's final commit to reproduce them.
No clinical data, external inference API, gated weights, fine-tuning, or clinician
review was used. This host has CPU PyTorch; real inference throughput/VRAM was not tested.

## Reproduction

From the repository root after installing dependencies:

```bash
python -m pytest -q
python scripts/run_comparison.py --mock --n-samples 10
python -m compileall -q src scripts tests
```

Mock mode is synthetic regardless of whether a local dataset happens to exist.
Real mode requires explicit, readable inputs and a valid requested split. Successful
runs record input and valid counts; input validation fails the entire requested run
instead of silently excluding invalid cases. Therefore successful runs have zero
excluded inputs. Inference failures prevent aggregate scoring.

## Data and privacy status

MIMIC-CXR is a credentialed-access resource. The source is
[PhysioNet MIMIC-CXR v2.1.0](https://physionet.org/content/mimic-cxr/2.1.0/).
The repository alone does not prove access authorization. The previous Kaggle link
has been removed from the recommended workflow. No restricted data was obtained in
this audit. Real outputs are git-ignored by default and must remain private unless
publication is permitted. Public artifacts should contain reviewed aggregates only.

The current-file cleanup is not Git-history erasure. Historical patient-level
artifacts remain accessible in prior commits; repository-history cleanup and
assessment of any prior external processing remain outstanding. No secrets were
intentionally accessed or added; this is not a comprehensive historical secret scan.

## What still requires a real experiment

1. Confirm dataset authorization and official splits; prepare accurate local metadata.
2. Obtain authorized model access and sufficient compute; pin model revisions.
3. Rebuild a patient-disjoint training index and verify real model loading/precision.
4. Manually verify evaluation answers and references; define grounding/error review.
5. Run zero-shot versus retrieval on the same cases, then publish permitted aggregates.
6. Confirm personal/team contributions before first-person CV attribution.

No positive accuracy, model superiority, fine-tuning gain or hallucination reduction
claim is supported by this audit. That does not invalidate the implemented prototype;
it defines precisely what has and has not been demonstrated.
