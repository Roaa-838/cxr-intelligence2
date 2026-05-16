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


def load_qa_dataset(path: str, n: int) -> list[dict]:
    """Load QA dataset; fall back to generating synthetic pairs."""
    p = Path(path)
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        logger.info("Loaded %d QA pairs from %s", len(data), p)
        return data[:n]

    logger.warning("QA dataset not found at %s — generating mock pairs.", p)
    from src.data.dataset import SyntheticCXRDataset
    from scripts.generate_qa_dataset import generate_mock_qa

    ds = SyntheticCXRDataset(n_samples=min(n // 5, 50))
    reports = [
        {"study_id": ds[i].study_id, "text": ds[i].report,
         "image": ds[i].image_path}
        for i in range(len(ds))
    ]
    return generate_mock_qa(reports, n_questions=5)[:n]


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

    pipeline = RAGQAPipeline(
        retriever=retriever_name, generator="medgemma",
        cfg=cfg, mock=mock,
    ).load()

    # Load images and run QA
    batch = []
    for qa in qa_pairs:
        img = qa.get("image_path", qa.get("image", ""))
        if img and Path(img).exists():
            try:
                pil = PILImage.open(img).convert("RGB")
            except Exception:
                pil = PILImage.new("RGB", (448, 448), color=128)
        else:
            pil = PILImage.new("RGB", (448, 448), color=128)

        batch.append({
            "image":    pil,
            "question": qa["question"],
            "answer":   qa.get("answer", ""),
            "study_id": qa.get("study_id", ""),
        })

    results = pipeline.run_batch(batch, top_k=5)

    predictions = [r.get("answer", "") for r in results]
    references  = [b["answer"] for b in batch]
    questions   = [b["question"] for b in batch]
    q_types     = [qa.get("question_type", "general") for qa in qa_pairs]

    evaluator = QAEvaluator()
    metrics = evaluator.evaluate(
        predictions=predictions,
        references=references,
        questions=questions,
        question_types=q_types,
    )
    metrics["retriever"] = retriever_name

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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.config import load_config
    cfg = load_config(args.config)

    qa_pairs = load_qa_dataset(args.qa_dataset, args.n_samples)
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

    # Comparison
    if len(all_metrics) > 1:
        import pandas as pd
        print("\n" + "═" * 60)
        print("  QA RETRIEVER COMPARISON")
        print("═" * 60)
        rows = []
        for ret, m in all_metrics.items():
            rows.append({
                "Retriever":    ret,
                "BERTScore F1": round(m.get("bertscore_f1", 0), 4),
                "Token F1":     round(m.get("token_f1", 0), 4),
                "ROUGE-L":      round(m.get("rouge_l", 0), 4),
                "Exact Match":  round(m.get("exact_match", 0), 4),
            })
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        df.to_csv(output_dir / "qa_comparison.csv", index=False)
        print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
