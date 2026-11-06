#!/usr/bin/env python3
"""
scripts/evaluate_qa.py
========================
Evaluate the RAG-based QA system on the generated QA dataset.
Compares ColPali vs CLIP retrievers.

Usage:
    python scripts/evaluate_qa.py \
        --qa-dataset data/qa_dataset.json \
        --retrievers colpali clip \
        --n-samples 500

    # Mock mode
    python scripts/evaluate_qa.py --mock --n-samples 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


from src.evaluation.validation import (load_qa_pairs, prepare_qa_batch, require_mode,
                                       validate_index, validate_outputs)
from src.evaluation.provenance import runtime_metadata


def load_qa_dataset(path: str, n: int, mock=False):
    return load_qa_pairs(path, n, mock)


def run_qa_evaluation(
    retriever_name: str,
    qa_pairs: list[dict],
    cfg: dict,
    mock: bool,
    output_dir: Path,
) -> dict:
    """Run QA pipeline + evaluation for one retriever."""
    from PIL import Image as PILImage
    from src.pipelines.rag_qa import RAGQAPipeline
    from src.evaluation.qa_evaluator import QAEvaluator

    logger.info("─" * 55)
    logger.info("Evaluating retriever: %s (%d pairs)", retriever_name, len(qa_pairs))
    logger.info("─" * 55)

    require_mode(mock, [retriever_name], cfg)
    batch = prepare_qa_batch(qa_pairs, mock)
    pipeline = RAGQAPipeline(
        retriever=retriever_name, generator="medgemma",
        cfg=cfg, mock=mock,
    ).load()

    if not mock:
        validate_index(pipeline._retriever, qa_pairs)

    results = pipeline.run_batch(batch, top_k=5)

    validate_outputs(results, len(batch), "answer")
    predictions = [r.get("answer", "") for r in results]
    references  = [b["answer"] for b in batch]
    questions   = [b["question"] for b in batch]
    q_types     = [qa.get("question_type", "general") for qa in qa_pairs]

    evaluator = QAEvaluator(use_bertscore=not mock)
    metrics = evaluator.evaluate(
        predictions=predictions,
        references=references,
        questions=questions,
        question_types=q_types,
    )
    metrics["retriever"] = retriever_name
    metrics["evaluation_mode"] = "mock" if mock else "real"
    metrics["n_input"] = len(batch)
    metrics["n_valid_images"] = len(batch) if not mock else 0
    metrics["n_valid"] = len(results)
    metrics["n_excluded"] = 0
    metrics["runtime"] = runtime_metadata(pipeline)

    evaluator.print_results(metrics, title=f"QA Results — {retriever_name}")

    # Save per-sample
    per_sample = [
        {
            "study_id":   r.get("study_id", ""),
            "question":   r.get("question", ""),
            "prediction": r.get("answer", ""),
            "reference":  b["answer"],
            "question_type": qa.get("question_type", "general"),
            "latency_s":  r.get("latency_s", 0),
        }
        for r, b, qa in zip(results, batch, qa_pairs)
    ]
    out_file = output_dir / f"qa_eval_{retriever_name}.json"
    with open(out_file, "w") as f:
        json.dump({"metrics": metrics, "samples": per_sample}, f, indent=2)
    logger.info("Saved -> %s", out_file)

    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RAG QA pipeline")
    p.add_argument("--qa-dataset",  default="data/qa_dataset.json")
    p.add_argument("--output-dir",  default="results")
    p.add_argument("--retrievers",  nargs="+", default=["colpali", "clip"],
                   choices=["colpali", "clip", "mock"])
    p.add_argument("--n-samples",   type=int, default=500)
    p.add_argument("--mock",        action="store_true")
    p.add_argument("--config",      default="configs/default.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) / ("mock" if args.mock else "real")
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.config import load_config
    cfg = load_config(args.config)
    require_mode(args.mock, args.retrievers, cfg)

    qa_pairs = load_qa_dataset(args.qa_dataset, args.n_samples, args.mock)
    logger.info("QA pairs to evaluate: %d", len(qa_pairs))

    all_metrics: dict[str, dict] = {}
    retrievers = ["mock"] if args.mock else args.retrievers

    for retriever in retrievers:
        try:
            metrics = run_qa_evaluation(
                retriever_name=retriever,
                qa_pairs=qa_pairs,
                cfg=cfg,
                mock=args.mock,
                output_dir=output_dir,
            )
            all_metrics[retriever] = metrics
        except Exception as e:
            logger.error("QA evaluation failed for %s: %s", retriever, e, exc_info=True)
            all_metrics[retriever] = {"status": "failed", "error": str(e)}

    with open(output_dir / "qa_summary.json", "w", encoding="utf-8") as f:
        json.dump({"evaluation_mode": "mock" if args.mock else "real", "retrievers": all_metrics}, f, indent=2)
    if any("error" in m for m in all_metrics.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
