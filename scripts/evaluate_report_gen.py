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


def _is_valid_path(val) -> bool:
    """Return True if val is a non-null, non-'nan' non-empty string path."""
    if val is None:
        return False
    import math
    if isinstance(val, float) and math.isnan(val):
        return False
    s = str(val).strip()
    return bool(s) and s.lower() != "nan"


def _open_image(path_str: str):
    """Open an image at an absolute path. Returns PIL Image or None."""
    from PIL import Image as _PIL
    p = Path(path_str)
    if not p.exists() or not p.is_file():
        return None
    try:
        return _PIL.open(p).convert("RGB")
    except Exception as e:
        logger.debug("Could not open %s: %s", p, e)
        return None


def load_test_samples(data_dir: Path, n: int, split: str = "test") -> list[dict]:
    """
    Load n samples from metadata.csv for evaluation.

    Image resolution priority (real MIMIC-CXR dataset):
      1. image_path column  — absolute path written by download_data.py
      2. image column       — may also be absolute or relative
      3. Skip gracefully    — sample included with image=None so report
                              text metrics still run

    NaN handling: pandas reads empty strings back as NaN after a CSV
    round-trip. We check for NaN/nan explicitly before Path operations.
    """
    csv = data_dir / "metadata.csv"
    if not csv.exists():
        logger.warning("metadata.csv not found — using synthetic data.")
        from src.data.dataset import SyntheticCXRDataset
        ds = SyntheticCXRDataset(n_samples=n)
        return [
            {"study_id": s.study_id, "image": s.image, "report": s.report}
            for s in [ds[i] for i in range(len(ds))]
        ]

    df = pd.read_csv(csv, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Filter by split; fall back to validate if test is empty
    if "split" in df.columns:
        subset = df[df["split"] == split]
        if subset.empty and split == "test":
            logger.warning(
                "No rows with split='test'. Falling back to split='validate'."
            )
            subset = df[df["split"] == "validate"]
        df = subset

    # Drop rows with no report text, then take first n that have an image
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().str.len() > 10]

    # Prefer rows that have a resolvable image, but never skip all
    has_img_col  = "image_path" in df.columns or "image" in df.columns
    samples      = []
    skipped_img  = 0

    for _, row in df.iterrows():
        if len(samples) >= n:
            break

        report = str(row.get("text", "")).strip()
        if not report:
            continue

        # Resolve image — try image_path first, then image column
        pil = None
        for col in ("image_path", "image"):
            val = row.get(col)
            if _is_valid_path(val):
                pil = _open_image(str(val))
                if pil is not None:
                    break

        if pil is None:
            skipped_img += 1

        samples.append({
            "study_id": str(row.get("study_id", f"row_{len(samples)}")),
            "image":    pil,      # PIL Image or None (metrics still run on text)
            "report":   report,
        })

    loaded_imgs = sum(1 for s in samples if s["image"] is not None)
    logger.info(
        "Loaded %d samples: %d with images, %d text-only (image missing/unresolved).",
        len(samples), loaded_imgs, skipped_img,
    )
    if loaded_imgs == 0:
        logger.warning(
            "No images loaded! MedGemma will receive blank placeholder images. "
            "Check that image_path values in metadata.csv point to existing files. "
            "Run: python scripts/download_data.py --explore"
        )
    return samples


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

    pipeline = ReportGenerationPipeline(
        model=model_name, cfg=cfg, mock=mock
    ).load()

    evaluator = ReportEvaluator(use_radgraph=True)

    # Run inference
    results = pipeline.run_batch(samples, show_progress=True)

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
        ("bertscore_f1", "BERTScore F1"),
        ("rouge_l",      "ROUGE-L"),
        ("rouge_1",      "ROUGE-1"),
        ("bleu_4",       "BLEU-4"),
        ("bleu_1",       "BLEU-1"),
    ]
    rows = []
    for model, metrics in all_metrics.items():
        row = {"Model": model}
        for key, label in metric_keys:
            row[label] = round(metrics.get(key, 0.0), 4)
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    from src.config import load_config
    cfg = load_config(args.config)

    # Load test samples
    logger.info("Loading %d test samples …", args.n_samples)
    samples = load_test_samples(
        data_dir=Path(args.data_dir),
        n=args.n_samples,
        split=args.split,
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
    summary = {"models": all_metrics, "n_samples": args.n_samples, "split": args.split}
    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Evaluation complete. Summary -> %s/evaluation_summary.json", output_dir)


if __name__ == "__main__":
    main()
