#!/usr/bin/env python3
"""
scripts/evaluate_report_gen.py
================================
Evaluate and compare report generation models on the MIMIC-CXR test set.

Compares:
  A. MedGemma-4B-it (zero-shot)
  B. MedGemma-4B-it (fine-tuned, if checkpoint available)
  C. CLIP-RAG + MedGemma

Metrics: BLEU-4, ROUGE-L, BERTScore F1, RadGraph F1

Usage:
    # Compare all models on 500 test samples
    python scripts/evaluate_report_gen.py --n-samples 500

    # Single model evaluation
    python scripts/evaluate_report_gen.py --models medgemma --n-samples 100

    # Mock mode (no GPU — sanity check)
    python scripts/evaluate_report_gen.py --mock --n-samples 10
"""

import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Windows: force UTF-8 on stdout/stderr so Unicode log chars don't crash ──
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

Path("results").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/eval_report_gen.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


from src.evaluation.provenance import runtime_metadata
from src.evaluation.validation import (load_report_samples, require_mode, validate_reports,
                                       validate_index, validate_outputs)


def load_test_samples(data_dir: Path, n: int, split: str = "test", mock=False):
    return load_report_samples(data_dir, n, split, mock)


def run_model_evaluation(
    model_name: str,
    samples: list[dict],
    cfg: dict,
    mock: bool,
    output_dir: Path,
) -> dict:
    """Run evaluation for one model configuration."""
    logger.info("─" * 55)
    logger.info("Evaluating model: %s (%d samples)", model_name, len(samples))
    logger.info("─" * 55)

    from src.pipelines.report_generation import ReportGenerationPipeline
    from src.evaluation.metrics import ReportEvaluator

    require_mode(mock, [model_name], cfg)
    validate_reports(samples)
    pipeline = ReportGenerationPipeline(
        model=model_name, cfg=cfg, mock=mock
    ).load()

    if not mock and pipeline._retriever is not None:
        validate_index(pipeline._retriever, samples)
    evaluator = ReportEvaluator(use_radgraph=not mock, use_bertscore=not mock)

    # Run inference
    results = pipeline.run_batch(samples, show_progress=True)

    validate_outputs(results, len(samples), "raw")

    # Extract generated and reference texts
    generated = [r.get("raw", r.get("findings", "") + "\n" + r.get("impression", ""))
                 for r in results]
    references = [s.get("report", "") for s in samples]

    # Filter out empty results
    pairs = [(g, r) for g, r in zip(generated, references) if g.strip() and r.strip()]
    if not pairs:
        logger.warning("No valid pairs to evaluate.")
        return {}

    gen_texts, ref_texts = zip(*pairs)

    # Compute metrics
    metrics = evaluator.evaluate(list(gen_texts), list(ref_texts))
    metrics["model"]    = model_name
    metrics["evaluation_mode"] = "mock" if mock else "real"
    metrics["n_input"] = len(samples)
    metrics["n_valid_images"] = 0 if mock else len(samples)
    metrics["n_excluded"] = 0
    metrics["runtime"] = runtime_metadata(pipeline)
    metrics["n_valid"]  = len(pairs)
    metrics["n_errors"] = sum(1 for r in results if r.get("error", ""))

    evaluator.print_results(metrics, title=f"Results — {model_name}")

    # Save per-sample outputs
    out_file = output_dir / f"report_gen_{model_name.replace('-', '_')}.json"
    per_sample = [
        {
            "study_id":   r.get("study_id", ""),
            "generated":  r.get("raw", ""),
            "findings":   r.get("findings", ""),
            "impression": r.get("impression", ""),
            "reference":  s.get("report", ""),
            "latency_s":  r.get("latency_s", 0),
        }
        for r, s in zip(results, samples)
    ]
    with open(out_file, "w") as f:
        json.dump({"metrics": metrics, "samples": per_sample}, f, indent=2)
    logger.info("Saved -> %s", out_file)

    return metrics


def build_comparison_table(all_metrics: dict[str, dict]) -> pd.DataFrame:
    """Format all model results into a comparison DataFrame."""
    metric_keys = [
        ("radgraph_f1",  "RadGraph F1"),
        ("clinical_entity_overlap_f1", "Clinical entity-overlap heuristic F1"),
        ("bertscore_f1", "BERTScore F1"),
        ("rouge_l",      "ROUGE-L"),
        ("rouge_1",      "ROUGE-1"),
        ("bleu_4",       "BLEU-4"),
        ("bleu_1",       "BLEU-1"),
    ]
    rows = []
    for model, metrics in all_metrics.items():
        row = {"Model": model, "evaluation_mode": metrics.get("evaluation_mode"), "status": metrics.get("status", "completed")}
        for key, label in metric_keys:
            value = metrics.get(key)
            row[label] = round(value, 4) if isinstance(value, (int, float)) else None
        row["N Valid"] = metrics.get("n_valid", 0)
        row["N Errors"] = metrics.get("n_errors", 0)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate report generation models")
    p.add_argument("--data-dir",   default="data/processed",      help="Processed data dir")
    p.add_argument("--output-dir", default="results",             help="Output directory")
    p.add_argument("--models",     nargs="+",
                   default=["medgemma", "clip-rag"],
                   choices=["medgemma", "medgemma-ft", "clip-rag", "mock"],
                   help="Models to evaluate")
    p.add_argument("--n-samples",  type=int, default=500,         help="Test samples per model")
    p.add_argument("--split",      default="test",                help="Dataset split")
    p.add_argument("--mock",       action="store_true",           help="Mock mode (no GPU)")
    p.add_argument("--config",     default="configs/default.yaml",help="Config YAML path")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) / ("mock" if args.mock else "real")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    from src.config import load_config
    cfg = load_config(args.config)
    require_mode(args.mock, args.models, cfg)

    # Load test samples
    logger.info("Loading %d test samples …", args.n_samples)
    samples = load_test_samples(
        data_dir=Path(args.data_dir),
        n=args.n_samples,
        split=args.split,
        mock=args.mock,
    )
    logger.info("Loaded %d samples.", len(samples))

    # Evaluate each model
    all_metrics: dict[str, dict] = {}
    models_to_eval = ["mock"] if args.mock else args.models

    for model_name in models_to_eval:
        try:
            metrics = run_model_evaluation(
                model_name=model_name,
                samples=samples,
                cfg=cfg,
                mock=args.mock,
                output_dir=output_dir,
            )
            all_metrics[model_name] = metrics
        except Exception as e:
            logger.error("Evaluation failed for %s: %s", model_name, e, exc_info=True)
            all_metrics[model_name] = {"status": "failed", "error": str(e), "evaluation_mode": "mock" if args.mock else "real"}

    # Comparison table
    if len(all_metrics) > 1:
        print("\n" + "═" * 70)
        print("  MODEL COMPARISON TABLE")
        print("═" * 70)
        comparison_df = build_comparison_table(all_metrics)
        print(comparison_df.to_string(index=False))
        comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
        print(f"\n  Saved -> {output_dir}/model_comparison.csv")
        print("═" * 70 + "\n")

    # Save summary JSON
    summary = {"models": all_metrics, "n_requested": args.n_samples, "split": args.split, "evaluation_mode": "mock" if args.mock else "real"}
    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Evaluation complete. Summary -> %s/evaluation_summary.json", output_dir)

    if any("error" in m for m in all_metrics.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
