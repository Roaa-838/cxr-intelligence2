# QA annotation protocol - planned real evaluation

The repository has code for generating candidate QA pairs from reports. This does
not establish a completed 2,500-question benchmark or expert clinical annotations.
Historical mock questions can confuse mention of a finding with its presence;
they are software fixtures, not gold-standard medical labels.

For a real evaluation:

1. Obtain authorized data and retain official study/patient IDs privately.
2. Keep evaluation patients separate from the training retrieval index and any fine-tuning data.
3. Draft questions locally, or only with a provider arrangement explicitly permitted for that data.
   The existing generation scripts can call external services; do not use those paths for restricted
   reports without separately verifying authorization. This audit did not call a provider.
4. Manually verify question, answer, negation, laterality and report grounding.
   Record the annotator role and protocol; synthetic drafts are not clinician annotations.
5. Store a private JSON list with `study_id`, `subject_id`, `image_path`, `question`,
   `answer` and optionally `question_type`. Image paths are relative to the JSON file or absolute.
   Add patient IDs from the source metadata; legacy generation outputs may not carry them.
6. Use the same verified questions and generation settings for each retriever.
7. Publish permitted aggregates and methodology only, not restricted reports or patient identifiers.

The strict evaluator checks structural validity and train-index overlap; it cannot
establish medical label correctness, authorization, or the honesty of supplied provenance.

Source: [MIMIC-CXR, PhysioNet](https://physionet.org/content/mimic-cxr/2.1.0/).
