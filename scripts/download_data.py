#!/usr/bin/env python3
"""
scripts/download_data.py  (REAL DATA VERSION)
==============================================
Downloads the real MIMIC-CXR dataset from Kaggle and prepares it.

Dataset: https://www.kaggle.com/datasets/simhadrisadaram/mimic-cxr-dataset

SETUP (one-time):
    1. Go to https://www.kaggle.com/settings -> API -> Create New Token
    2. Save kaggle.json to C:\\Users\\<YOU>\\.kaggle\\kaggle.json
       OR set env vars:  KAGGLE_USERNAME  and  KAGGLE_KEY  in .env

USAGE:
    python scripts/download_data.py                  # download + process all
    python scripts/download_data.py --explore        # inspect structure only
    python scripts/download_data.py --n-train 5000   # use 5K training samples
    python scripts/download_data.py --no-download    # skip download (already done)
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image as PILImage

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
        logging.FileHandler("results/download.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

RAW_DIR       = Path("data/mimic-cxr-raw")
PROCESSED_DIR = Path("data/processed")
KAGGLE_SLUG   = "simhadrisadaram/mimic-cxr-dataset"

REPORT_COLS = ["text","report","report_text","radiology_report","findings","impression","clinical_notes","note"]
IMAGE_COLS  = ["image","image_path","filepath","path","img","img_path","filename","jpg","dicom","file"]
STUDY_COLS  = ["study_id","study","studyid","id"]


def download_from_kaggle(dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    existing = list(dest.rglob("*.csv")) + list(dest.rglob("*.jpg")) + list(dest.rglob("*.png"))
    if existing:
        logger.info("Found %d existing files in %s -- skipping download.", len(existing), dest)
        return True
    try:
        import kaggle
        logger.info("Downloading %s ...", KAGGLE_SLUG)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(KAGGLE_SLUG, path=str(dest), unzip=True, quiet=False)
        logger.info("Download complete -> %s", dest)
        return True
    except ImportError:
        logger.error("kaggle not installed: pip install kaggle")
    except Exception as e:
        logger.error("Kaggle download failed: %s", e)
    logger.info("Manual steps: kaggle.com/datasets/%s -> Download -> extract to %s", KAGGLE_SLUG, dest)
    return False


def explore_raw_dir(raw_dir: Path) -> dict:
    all_f = list(raw_dir.rglob("*"))
    csvs  = [f for f in all_f if f.suffix == ".csv"]
    imgs  = [f for f in all_f if f.suffix.lower() in (".jpg",".jpeg",".png")]
    logger.info("")
    logger.info("=== RAW DATASET STRUCTURE ===")
    logger.info("  CSV files:    %d", len(csvs))
    logger.info("  Image files:  %d", len(imgs))
    for d in sorted(raw_dir.iterdir())[:8]:
        logger.info("  [%s] %s", "DIR" if d.is_dir() else "FILE", d.name)
    for csv in csvs[:5]:
        try:
            df = pd.read_csv(csv, nrows=2)
            logger.info("  CSV: %-40s  cols=%s", csv.relative_to(raw_dir), list(df.columns[:6]))
        except Exception:
            pass
    return {"csvs": csvs, "imgs": imgs}


def find_best_csv(csvs: list[Path]) -> Path | None:
    best, best_score = None, -1
    for csv in csvs:
        try:
            df  = pd.read_csv(csv, nrows=3)
            low = [c.strip().lower() for c in df.columns]
            score = 0
            if any("text" in c or "report" in c for c in low): score += 10
            if any("study" in c for c in low):                 score += 5
            if any("image" in c or "path" in c for c in low):  score += 3
            if any("split" in c for c in low):                  score += 2
            if score > best_score:
                best_score, best = score, csv
        except Exception:
            pass
    if best:
        logger.info("Best CSV: %s (score=%d)", best.name, best_score)
        df = pd.read_csv(best, nrows=2)
        logger.info("Columns: %s", list(df.columns))
    return best


def normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
    for c in REPORT_COLS:
        if c in df.columns and "text" not in df.columns:
            df = df.rename(columns={c: "text"}); break
    for c in IMAGE_COLS:
        if c in df.columns and "image" not in df.columns:
            df = df.rename(columns={c: "image"}); break
    for c in STUDY_COLS:
        if c in df.columns and "study_id" not in df.columns:
            df = df.rename(columns={c: "study_id"}); break
    return df


def build_image_lookup(raw_dir: Path) -> dict[str, Path]:
    logger.info("Scanning for images ...")
    lookup: dict[str, Path] = {}
    for img in raw_dir.rglob("*"):
        if img.suffix.lower() not in (".jpg",".jpeg",".png",".dicom",".dcm"):
            continue
        lookup[img.stem] = img
        lookup[img.name] = img
        parts = img.parts
        if len(parts) >= 2:
            lookup[f"{parts[-2]}/{img.name}"] = img
    logger.info("Image lookup: %d keys -> %d unique images", len(lookup), len(set(lookup.values())))
    return lookup


def resolve_image(val: str, raw_dir: Path, lookup: dict) -> Path | None:
    if not val or (isinstance(val, float)):
        return None
    val = str(val).strip()
    for cand in [Path(val), raw_dir / val, raw_dir / Path(val).name]:
        if cand.exists():
            return cand
    for key in [val, Path(val).name, Path(val).stem]:
        if key in lookup:
            return lookup[key]
    return None


def build_processed_csv(best_csv, raw_dir, output_dir, n_train, n_test):
    logger.info("")
    logger.info("=== BUILDING PROCESSED DATASET ===")
    df = pd.read_csv(best_csv)
    logger.info("Loaded %d rows from %s", len(df), best_csv.name)
    df = normalise_df(df)

    if "text" not in df.columns:
        logger.error("No report column. Columns: %s", list(df.columns))
        logger.error("Add your column name to REPORT_COLS in download_data.py")
        raise ValueError("No report column found")

    df = df[df["text"].notna() & (df["text"].str.strip().str.len() > 20)]
    logger.info("After filter: %d rows with non-empty reports", len(df))

    lookup = build_image_lookup(raw_dir)
    image_col = "image" if "image" in df.columns else None

    resolved = []
    missing  = 0
    for _, row in df.iterrows():
        val  = str(row.get(image_col, "")) if image_col else ""
        path = resolve_image(val, raw_dir, lookup)
        resolved.append(str(path.resolve()) if path else "")
        if not path:
            missing += 1

    df["image"]      = resolved
    df["image_path"] = resolved

    n_with = sum(1 for p in resolved if p)
    logger.info("Images resolved: %d / %d  (missing=%d)", n_with, len(df), missing)

    if missing > len(df) * 0.5:
        logger.warning(
            "Over 50%% of images could not be resolved. "
            "Check that image filenames in the CSV match actual files in %s", raw_dir
        )

    if "split" not in df.columns:
        logger.info("Creating 80/10/10 train/validate/test split ...")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(df)
        df["split"] = "train"
        df.iloc[int(n*0.8):int(n*0.9), df.columns.get_loc("split")] = "validate"
        df.iloc[int(n*0.9):,           df.columns.get_loc("split")] = "test"

    for spl, grp in df.groupby("split"):
        with_img = (grp["image"] != "").sum()
        logger.info("  %-12s %5d rows | %d with images", spl, len(grp), with_img)

    if n_train or n_test:
        parts = []
        for spl, grp in df.groupby("split"):
            limit = {"train": n_train, "validate": n_train, "test": n_test}.get(spl)
            if limit and len(grp) > limit:
                grp = grp.sample(n=limit, random_state=42)
            parts.append(grp)
        df = pd.concat(parts).reset_index(drop=True)
        logger.info("After subsampling: %d rows", len(df))

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "metadata.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    logger.info("Saved -> %s", out)
    return out


def print_summary(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    logger.info("")
    logger.info("=" * 55)
    logger.info("  DATASET READY  |  %s", csv_path)
    logger.info("=" * 55)
    logger.info("  Total rows: %d", len(df))
    df.columns = [c.lower() for c in df.columns]
    if "split" in df.columns:
        for spl, g in df.groupby("split"):
            wi = (g.get("image","") != "").sum()
            logger.info("    %-12s %d rows | %d images", spl, len(g), wi)
    if "text" in df.columns:
        l = df["text"].dropna().str.len()
        logger.info("  Report chars: mean=%.0f  min=%d  max=%d", l.mean(), l.min(), l.max())
        sample = df["text"].dropna().iloc[0][:250]
        logger.info("  Sample: %s ...", sample.replace("\n"," "))
    logger.info("")
    logger.info("  NEXT STEPS:")
    logger.info("    python scripts/build_index.py --model clip --n-samples 5000")
    logger.info("    python scripts/evaluate_report_gen.py --n-samples 100")
    logger.info("=" * 55)


def _create_synthetic(output_dir: Path, n: int = 50) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / "images"
    img_dir.mkdir(exist_ok=True)
    REPORTS = [
        ("Heart normal. Lungs clear. No effusion or pneumothorax.", "No acute process."),
        ("Mild cardiomegaly. Bibasilar atelectasis. Small left effusion.", "Cardiomegaly with atelectasis."),
        ("Right lower lobe consolidation. Heart normal.", "Right lower lobe pneumonia."),
        ("Hyperinflated lungs. Flat diaphragms.", "COPD pattern. No acute process."),
        ("Small left apical pneumothorax.", "Left apical pneumothorax."),
    ]
    rows = []
    for i in range(n):
        f, imp = REPORTS[i % len(REPORTS)]
        rep = f"FINDINGS:\n{f}\n\nIMPRESSION:\n{imp}"
        img_path = img_dir / f"sample_{i:04d}.jpg"
        arr = np.random.randint(30, 210, (448, 448), dtype=np.uint8)
        PILImage.fromarray(arr).convert("RGB").save(str(img_path))
        rows.append({
            "study_id": f"synth_{i:05d}", "subject_id": f"p{i%5:05d}",
            "image": str(img_path.resolve()), "image_path": str(img_path.resolve()),
            "text": rep,
            "split": ["train","train","train","validate","test"][i % 5],
        })
    pd.DataFrame(rows).to_csv(output_dir / "metadata.csv", index=False)
    logger.info("Synthetic dataset: %d samples -> %s", n, output_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir",     default="data/mimic-cxr-raw")
    p.add_argument("--output-dir",  default="data/processed")
    p.add_argument("--n-train",     type=int, default=None)
    p.add_argument("--n-test",      type=int, default=None)
    p.add_argument("--explore",     action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--synthetic",   action="store_true")
    p.add_argument("--n-synthetic", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()

    if args.synthetic:
        _create_synthetic(Path(args.output_dir), n=args.n_synthetic)
        return

    raw_dir    = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    if not args.no_download:
        ok = download_from_kaggle(raw_dir)
        if not ok:
            sys.exit(1)

    info     = explore_raw_dir(raw_dir)
    best_csv = find_best_csv(info["csvs"])

    if args.explore or best_csv is None:
        if best_csv is None:
            logger.error("No usable CSV found in %s", raw_dir)
        return

    processed_csv = build_processed_csv(
        best_csv=best_csv, raw_dir=raw_dir, output_dir=output_dir,
        n_train=args.n_train, n_test=args.n_test,
    )
    print_summary(processed_csv)


if __name__ == "__main__":
    main()
