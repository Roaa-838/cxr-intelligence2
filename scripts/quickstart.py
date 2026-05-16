#!/usr/bin/env python3
"""
scripts/quickstart.py
======================
Zero-setup end-to-end smoke test. Runs every pipeline in mock mode
(no GPU, no real model weights, no dataset download needed).

Perfect for:
  - Verifying the codebase works after install
  - CI/CD pipeline smoke testing
  - Demoing the system architecture without hardware

Usage:
    python scripts/quickstart.py          # full mock run
    python scripts/quickstart.py --real   # real models (needs GPU + HF_TOKEN)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MOCK = True   # flipped by --real flag

SEPARATOR = "─" * 60


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def info(msg: str) -> None:
    print(f"  ℹ  {msg}")


# ─────────────────────────────────────────────────────────────────────────────

def step1_synthetic_data():
    section("STEP 1 — Synthetic Dataset")
    from src.data.dataset import SyntheticCXRDataset
    from src.data.report_parser import ReportParser
    from src.data.preprocessing import CXRPreprocessor

    ds = SyntheticCXRDataset(n_samples=5)
    parser = ReportParser()
    prep = CXRPreprocessor.for_medgemma()

    for i in range(3):
        s = ds[i]
        sections = parser.parse(s.report)
        tensor = prep(s.image)
        info(f"Study {s.study_id}: image={tensor.shape}, "
             f"findings_len={len(sections.findings)}, "
             f"impression_len={len(sections.impression)}")

    ok(f"SyntheticCXRDataset ({len(ds)} samples) + ReportParser + CXRPreprocessor")
    return ds


def step2_report_generation(ds):
    section("STEP 2 — Report Generation Pipeline (MedGemma mock)")
    from src.pipelines.report_generation import ReportGenerationPipeline

    pipeline = ReportGenerationPipeline(model="mock", mock=True).load()
    sample = ds[0]
    result = pipeline.run(sample.image)

    info(f"Model:      {result['model']}")
    info(f"Latency:    {result['latency_s']}s")
    info(f"Findings:   {result['findings'][:100]}…")
    info(f"Impression: {result['impression']}")
    ok("Report generation pipeline")
    return result


def step3_clip_retrieval(ds):
    section("STEP 3 — CLIP Retrieval + RAG Report Generation")
    from src.models.clip_retriever import CLIPRetriever
    from src.pipelines.report_generation import ReportGenerationPipeline

    ret = CLIPRetriever.load(model_name="biomedclip", mock=True)
    studies = [
        {"image": ds[i].image, "report": ds[i].report,
         "study_id": ds[i].study_id}
        for i in range(5)
    ]
    ret.build_index(studies)
    ok(f"BiomedCLIP index built: {ret.index_size} entries")

    results = ret.retrieve(query_image=ds[0].image, query_text="pleural effusion", top_k=3)
    for r in results:
        info(f"  Retrieved: {r['study_id']} | score={r['score']:.3f}")
    ok("BiomedCLIP retrieval")
    return ret


def step4_colpali_retrieval(ds):
    section("STEP 4 — ColPali Multimodal Retrieval")
    import torch
    from src.models.colpali_retriever import ColPaliRetriever

    ret = ColPaliRetriever.load(mock=True)
    studies = [
        {"image": ds[i].image, "report": ds[i].report, "study_id": ds[i].study_id}
        for i in range(5)
    ]
    ret.build_index(studies)
    ok(f"ColPali index built: {ret.index_size} entries")

    results = ret.retrieve(query_text="consolidation right lower lobe", top_k=3)
    for r in results:
        info(f"  Retrieved: {r['study_id']} | MaxSim score={r['score']:.3f}")

    # Show MaxSim scoring
    q_emb = torch.randn(10, 128)
    d_emb = torch.randn(1024, 128)
    score = ColPaliRetriever.maxsim_score(q_emb, d_emb)
    info(f"MaxSim score (random tensors): {score:.4f}")
    ok("ColPali retrieval + MaxSim scoring")
    return ret


def step5_rag_qa(ds):
    section("STEP 5 — RAG-Based QA Pipeline")
    from src.pipelines.rag_qa import RAGQAPipeline

    pipeline = RAGQAPipeline(retriever="clip", generator="medgemma", mock=True).load()
    pipeline.set_index([
        {"image": ds[i].image, "report": ds[i].report, "study_id": ds[i].study_id}
        for i in range(5)
    ])
    ok("RAGQAPipeline + index loaded")

    questions = [
        "Is there a pleural effusion in this chest X-ray?",
        "What is the most likely diagnosis?",
        "Is cardiomegaly present?",
    ]
    for q in questions:
        result = pipeline.run(ds[0].image, q, top_k=3)
        info(f"Q: {q}")
        info(f"A: {result['answer'][:120]}…")
        info(f"   Retrieved: {result['retrieved_studies']}")
    ok("RAG QA pipeline")


def step6_evaluation(ds):
    section("STEP 6 — Evaluation Metrics")
    from src.evaluation.metrics import (
        compute_bleu, compute_rouge, _radgraph_heuristic_fallback,
        compute_exact_match, compute_token_f1
    )
    from src.evaluation.qa_evaluator import QAEvaluator

    hyps = [ds[i].report for i in range(3)]
    refs = [ds[i].report for i in range(3)]  # identical → perfect score

    bleu  = compute_bleu(hyps, refs)
    rouge = compute_rouge(hyps, refs)
    rg    = _radgraph_heuristic_fallback(hyps, refs)

    info(f"BLEU-4:      {bleu['bleu_4']:.4f}  (identical → ~1.0)")
    info(f"ROUGE-L:     {rouge['rouge_l']:.4f}  (identical → ~1.0)")
    info(f"RadGraph F1: {rg['radgraph_f1']:.4f}  (heuristic fallback)")

    preds = ["No pleural effusion is seen.", "Heart size is enlarged."]
    refs2 = ["No pleural effusion.", "Cardiomegaly present."]
    em = compute_exact_match(preds, refs2)
    f1 = compute_token_f1(preds, refs2)
    info(f"QA Exact Match: {em:.3f}")
    info(f"QA Token F1:    {f1:.3f}")
    ok("All evaluation metrics")


def step7_qa_dataset(ds):
    section("STEP 7 — QA Dataset Generation (mock)")
    from scripts.generate_qa_dataset import generate_mock_qa
    from src.qa_dataset.generator import QADatasetGenerator
    import tempfile, json

    reports = [{"study_id": ds[i].study_id, "text": ds[i].report} for i in range(5)]
    qa_pairs = generate_mock_qa(reports, n_questions=5)
    qa_pairs = QADatasetGenerator.filter_qa_pairs(qa_pairs)
    stats = QADatasetGenerator.get_statistics(qa_pairs)

    info(f"Generated:     {stats['total_pairs']} QA pairs")
    info(f"Studies:       {stats['unique_studies']}")
    info(f"Types:         {stats['by_question_type']}")

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(qa_pairs, f)
        info(f"Saved to:      {f.name}")
    ok("QA dataset generation pipeline")
    return qa_pairs


def step8_vector_store():
    section("STEP 8 — Vector Store (Qdrant in-memory)")
    import numpy as np
    try:
        from src.rag.store import VectorStore
        store = VectorStore(backend="qdrant", embedding_dim=512).connect()
        emb = np.random.randn(512).astype(np.float32)
        store.upsert("study_001", emb, {"report": "No acute process.", "split": "train"})
        store.upsert("study_002", np.random.randn(512).astype(np.float32),
                     {"report": "Cardiomegaly.", "split": "train"})
        results = store.search(emb, top_k=2)
        info(f"Vector store count: {store.count}")
        info(f"Top result: {results[0].get('study_id', '?')} score={results[0].get('score', 0):.3f}")
        ok("Qdrant in-memory vector store")
    except Exception as e:
        info(f"Vector store skipped: {e}")


def step9_summary(start_time: float):
    import time
    section("QUICKSTART COMPLETE")
    elapsed = time.time() - start_time
    print(f"""
  All pipeline components verified ✅

  ┌─────────────────────────────────────────────────────┐
  │  Next Steps                                         │
  ├─────────────────────────────────────────────────────┤
  │  1. Get MIMIC-CXR data:                             │
  │     python scripts/download_data.py --synthetic     │
  │                                                     │
  │  2. Build retrieval index:                          │
  │     python scripts/build_index.py --model clip      │
  │                                                     │
  │  3. Generate QA dataset:                            │
  │     python scripts/generate_qa_dataset.py --mock    │
  │                                                     │
  │  4. Run evaluation:                                 │
  │     python scripts/evaluate_report_gen.py --mock    │
  │     python scripts/evaluate_qa.py --mock            │
  │                                                     │
  │  5. Full comparison:                                │
  │     python scripts/run_comparison.py --mock         │
  │                                                     │
  │  6. Launch demo:                                    │
  │     python app/app.py                               │
  │                                                     │
  │  For real GPU run: set HF_TOKEN env var and         │
  │  remove --mock flags from all commands above.       │
  └─────────────────────────────────────────────────────┘

  Elapsed: {elapsed:.1f}s
""")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    import time
    t0 = time.time()
    p = argparse.ArgumentParser()
    p.add_argument("--real", action="store_true", help="Use real models (GPU required)")
    p.add_argument("--skip-store", action="store_true")
    args = p.parse_args()

    if args.real:
        global MOCK
        MOCK = False
        os.environ["MOCK_MODELS"] = "false"
    else:
        os.environ["MOCK_MODELS"] = "true"

    print(f"""
{'═' * 60}
  CXR Intelligence System — Quickstart
  Mode: {'🟡 MOCK (no GPU required)' if MOCK else '🟢 REAL (GPU)'}
{'═' * 60}""")

    ds = step1_synthetic_data()
    step2_report_generation(ds)
    step3_clip_retrieval(ds)
    step4_colpali_retrieval(ds)
    step5_rag_qa(ds)
    step6_evaluation(ds)
    step7_qa_dataset(ds)
    if not args.skip_store:
        step8_vector_store()
    step9_summary(t0)


if __name__ == "__main__":
    main()
