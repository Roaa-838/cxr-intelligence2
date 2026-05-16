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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/build_index.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def _valid(val) -> bool:
    """True if val is a non-null, non-NaN, non-empty non-'nan' string."""
    if val is None:
        return False
    import math
    if isinstance(val, float) and math.isnan(val):
        return False
    return bool(str(val).strip()) and str(val).strip().lower() != "nan"


def load_training_studies(data_dir: Path, n: int, split: str = "train") -> list[dict]:
    """
    Load training studies for indexing.

    Uses the absolute image_path column written by download_data.py.
    Handles NaN values, skips missing images without crashing.
    Only includes rows that have a resolvable image (required for visual index).
    """
    csv = data_dir / "metadata.csv"
    if not csv.exists():
        logger.warning("metadata.csv not found — using synthetic data.")
        from src.data.dataset import SyntheticCXRDataset
        ds = SyntheticCXRDataset(n_samples=min(n, 100))
        return [
            {"study_id": s.study_id, "image": s.image,
             "image_path": s.image_path, "report": s.report}
            for s in [ds[i] for i in range(len(ds))]
        ]

    df = pd.read_csv(csv, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "split" in df.columns:
        df = df[df["split"] == split]

    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().str.len() > 10]

    studies = []
    skipped  = 0

    for _, row in df.iterrows():
        if len(studies) >= n:
            break

        # Resolve image — try image_path (absolute) then image column
        img_path = None
        for col in ("image_path", "image"):
            val = row.get(col)
            if _valid(val):
                p = Path(str(val))
                if p.exists():
                    img_path = p
                    break

        # Skip rows without a resolvable image — we need visuals for the index
        if img_path is None:
            skipped += 1
            continue

        try:
            from PIL import Image as _PIL
            pil = _PIL.open(img_path).convert("RGB")
        except Exception as e:
            logger.debug("Cannot open %s: %s", img_path, e)
            skipped += 1
            continue

        studies.append({
            "study_id":   str(row.get("study_id", f"study_{len(studies)}")),
            "image":      pil,
            "image_path": str(img_path),
            "report":     str(row.get("text", "")),
        })

    logger.info(
        "Loaded %d studies for indexing (%d skipped — missing image).",
        len(studies), skipped,
    )
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

    # Load PIL images from paths
    from PIL import Image
    for s in studies:
        if s.get("image") is None and s.get("image_path"):
            try:
                s["image"] = Image.open(s["image_path"]).convert("RGB")
            except Exception:
                s["image"] = Image.new("RGB", (448, 448), color=128)

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

    from PIL import Image
    for s in studies:
        if s.get("image") is None and s.get("image_path"):
            try:
                s["image"] = Image.open(s["image_path"]).convert("RGB")
            except Exception:
                s["image"] = Image.new("RGB", (224, 224), color=128)

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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load studies
    data_dir = Path(args.data_dir)
    if args.synthetic:
        from src.data.dataset import SyntheticCXRDataset
        n = min(args.n_samples, 100)
        ds = SyntheticCXRDataset(n_samples=n)
        studies = [
            {"study_id": ds[i].study_id, "image": ds[i].image,
             "image_path": "", "report": ds[i].report}
            for i in range(len(ds))
        ]
    else:
        studies = load_training_studies(data_dir, args.n_samples, args.split)

    logger.info("Loaded %d studies for indexing.", len(studies))

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
