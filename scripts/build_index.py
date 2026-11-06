#!/usr/bin/env python3
"""
scripts/build_index.py
========================
Build the ColPali or CLIP retrieval index from the MIMIC-CXR training set.
This index powers Mode 2 (RAG-based QA) and Mode 1 CLIP-RAG report generation.

Usage:
    # Build ColPali index (recommended for QA mode)
    python scripts/build_index.py --model colpali --n-samples 10000

    # Build CLIP/BiomedCLIP index (for comparison)
    python scripts/build_index.py --model clip --n-samples 10000

    # Build both
    python scripts/build_index.py --model both --n-samples 10000

    # Quick test with synthetic data
    python scripts/build_index.py --model clip --synthetic --n-samples 50
"""

import argparse
import io
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

Path("results").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/build_index.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


from src.evaluation.validation import load_report_samples, require_mode


def load_training_studies(data_dir: Path, n: int, split: str = "train"):
    if split != "train":
        raise ValueError("Research index must use the train split")
    studies = load_report_samples(data_dir, n, split)
    for s in studies:
        if not s.get("subject_id"):
            raise ValueError("subject_id is required to prevent patient leakage")
        s["data_mode"] = "real"
    return studies


def build_colpali_index(
    studies: list[dict],
    output_dir: Path,
    mock: bool = False,
) -> None:
    """Build and save ColPali patch-embedding index."""
    from src.models.colpali_retriever import ColPaliRetriever

    logger.info("Building ColPali index over %d studies …", len(studies))
    retriever = ColPaliRetriever.load(mock=mock)

    retriever.build_index(studies, batch_size=4)

    out_path = output_dir / "colpali_index"
    retriever.save_index(out_path)
    logger.info(
        "ColPali index saved → %s (.npy + .meta.pkl)\n"
        "  %d studies indexed, %.1f MB on disk",
        out_path,
        retriever.index_size,
        (Path(str(out_path) + ".npy")).stat().st_size / 1e6
        if (Path(str(out_path) + ".npy")).exists() else 0,
    )


def build_clip_index(
    studies: list[dict],
    output_dir: Path,
    model_name: str = "biomedclip",
    mock: bool = False,
) -> None:
    """Build and save CLIP/BiomedCLIP single-vector index."""
    from src.models.clip_retriever import CLIPRetriever

    logger.info("Building CLIP (%s) index over %d studies …", model_name, len(studies))
    retriever = CLIPRetriever.load(model_name=model_name, mock=mock)

    retriever.build_index(studies, batch_size=64)

    out_path = output_dir / "clip_index.pkl"
    retriever.save_index(out_path)
    logger.info("CLIP index saved -> %s (%d entries)", out_path, retriever.index_size)


def parse_args():
    p = argparse.ArgumentParser(description="Build retrieval index")
    p.add_argument("--model",       default="colpali",
                   choices=["colpali", "clip", "both"],
                   help="Which index to build")
    p.add_argument("--data-dir",    default="data/processed",   help="Processed data dir")
    p.add_argument("--output-dir",  default="data/index",       help="Index output dir")
    p.add_argument("--n-samples",   type=int, default=10000,    help="Training samples to index")
    p.add_argument("--split",       default="train",            help="Dataset split to index")
    p.add_argument("--clip-model",  default="biomedclip",
                   choices=["biomedclip", "clip-vit-l-14", "clip-vit-b-32"])
    p.add_argument("--mock",        action="store_true",        help="Mock mode (no GPU)")
    p.add_argument("--synthetic",   action="store_true",        help="Use synthetic data")
    p.add_argument("--config",      default="configs/default.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    require_mode(args.mock)
    if args.n_samples < 1:
        raise ValueError("n-samples must be positive")
    output_dir = Path(args.output_dir)
    if args.mock or args.synthetic:
        output_dir = output_dir / "synthetic_mock"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load studies
    data_dir = Path(args.data_dir)
    if args.synthetic or args.mock:
        studies = load_report_samples(data_dir, args.n_samples, split="train", mock=True)
        for s in studies:
            s["data_mode"] = "synthetic"
    else:
        studies = load_training_studies(data_dir, args.n_samples, args.split)

    logger.info("Loaded %d studies for indexing.", len(studies))

    if len(studies) == 0:
        logger.error(
            "0 studies with resolvable images. "
            "Run:  python scripts/download_data.py --no-download  "
            "to rebuild metadata.csv with correct image_path values, then retry."
        )
        sys.exit(1)

    if args.model in ("clip", "both"):
        build_clip_index(
            studies, output_dir,
            model_name=args.clip_model,
            mock=args.mock,
        )

    if args.model in ("colpali", "both"):
        build_colpali_index(studies, output_dir, mock=args.mock)

    logger.info("Indexing complete. Files in: %s", output_dir)
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / 1e6
        logger.info("  %s  (%.1f MB)", f.name, size_mb)


if __name__ == "__main__":
    main()
