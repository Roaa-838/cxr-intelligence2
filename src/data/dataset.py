"""src/data/dataset.py — MIMIC-CXR dataset loader."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class MIMICCXRSample:
    """One data point: a chest X-ray + associated report."""

    study_id: str
    subject_id: str
    image_path: str
    report: str                            # Full free-text report
    findings: str = ""                     # Parsed FINDINGS section
    impression: str = ""                   # Parsed IMPRESSION section
    split: str = "train"
    labels: dict[str, float] = field(default_factory=dict)  # CheXpert labels
    image: Optional[Image.Image] = None    # Lazy-loaded PIL image


class MIMICCXRDataset(Dataset):
    """
    PyTorch Dataset wrapping the MIMIC-CXR Kaggle export.

    Expected CSV columns (minimum):
        study_id, subject_id, image (path), text (report), split

    Optional label columns (CheXpert 14):
        Atelectasis, Cardiomegaly, Consolidation, Edema,
        Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity,
        No Finding, Pleural Effusion, Pleural Other, Pneumonia,
        Pneumothorax, Support Devices

    Args:
        csv_path:       Path to the metadata CSV (can be split-specific).
        image_root:     Root directory where image files live.
        split:          "train" | "validate" | "test" | None (all).
        transform:      Optional torchvision / albumentations transform.
        report_parser:  Optional callable to split report into sections.
        max_samples:    Limit dataset size (useful for quick experiments).
        preload_images: Cache all images in RAM on init (fast inference,
                        high RAM usage).
    """

    CHEXPERT_LABELS = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
        "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
        "Pneumonia", "Pneumothorax", "Support Devices",
    ]

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path = "",
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        report_parser: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        preload_images: bool = False,
    ) -> None:
        self.image_root = Path(image_root)
        self.transform = transform
        self.report_parser = report_parser
        self.preload_images = preload_images

        # ── Load metadata ────────────────────────────────────────────────────
        df = pd.read_csv(csv_path)
        logger.info("Loaded CSV: %s  (%d rows)", csv_path, len(df))

        # Normalise column names (handle different Kaggle export variants)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df = self._normalise_columns(df)

        # ── Filter by split ──────────────────────────────────────────────────
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].copy()
            logger.info("After split=%s filter: %d rows", split, len(df))

        # ── Drop rows without report ─────────────────────────────────────────
        df = df.dropna(subset=["text"])
        df = df[df["text"].str.strip() != ""]

        # ── Optional size limit ───────────────────────────────────────────────
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
            logger.info("Subsampled to %d rows", len(df))

        self.df = df.reset_index(drop=True)
        self._samples: list[Optional[MIMICCXRSample]] = [None] * len(self.df)

        if preload_images:
            self._preload_all_images()

        logger.info("Dataset ready: %d samples", len(self))

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Map common variant column names to canonical names."""
        rename = {}
        for col in df.columns:
            if col in ("image_path", "filepath", "path", "img_path"):
                rename[col] = "image"
            elif col in ("report", "report_text", "radiology_report"):
                rename[col] = "text"
        return df.rename(columns=rename) if rename else df

    def _build_sample(self, idx: int) -> MIMICCXRSample:
        row = self.df.iloc[idx]
        report = str(row.get("text", ""))

        # Parse report sections if parser provided
        findings, impression = "", ""
        if self.report_parser is not None:
            sections = self.report_parser(report)
            findings = sections.get("findings", "")
            impression = sections.get("impression", "")

        # CheXpert labels (NaN → 0)
        labels = {}
        for lbl in self.CHEXPERT_LABELS:
            lbl_lower = lbl.lower().replace(" ", "_")
            val = row.get(lbl, row.get(lbl_lower, float("nan")))
            labels[lbl] = float(val) if not pd.isna(val) else 0.0

        img_rel = str(row.get("image", ""))
        img_path = str(self.image_root / img_rel) if img_rel else ""

        sample = MIMICCXRSample(
            study_id=str(row.get("study_id", f"study_{idx}")),
            subject_id=str(row.get("subject_id", f"patient_{idx}")),
            image_path=img_path,
            report=report,
            findings=findings,
            impression=impression,
            split=str(row.get("split", "unknown")),
            labels=labels,
        )
        return sample

    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        if not image_path or not os.path.exists(image_path):
            return None
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as exc:
            logger.warning("Could not load image %s: %s", image_path, exc)
            return None

    def _preload_all_images(self) -> None:
        logger.info("Preloading %d images …", len(self))
        for i in range(len(self)):
            s = self[i]                     # triggers _build_sample + image load
            _ = s                           # already cached in self._samples

    # ── Public interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> MIMICCXRSample:
        if self._samples[idx] is None:
            self._samples[idx] = self._build_sample(idx)

        sample = self._samples[idx]

        # Lazy image loading
        if sample.image is None:
            sample.image = self._load_image(sample.image_path)

        # Apply transform if provided
        if self.transform is not None and sample.image is not None:
            sample.image = self.transform(sample.image)

        return sample

    def get_by_study_id(self, study_id: str) -> Optional[MIMICCXRSample]:
        matches = self.df[self.df["study_id"] == study_id]
        if matches.empty:
            return None
        return self[matches.index[0]]

    def get_label_distribution(self) -> pd.DataFrame:
        """Return frequency of each CheXpert label in this split."""
        rows = []
        for lbl in self.CHEXPERT_LABELS:
            lbl_lower = lbl.lower().replace(" ", "_")
            col = lbl if lbl in self.df.columns else lbl_lower
            if col in self.df.columns:
                pos = (self.df[col] == 1.0).sum()
                rows.append({"label": lbl, "positive": int(pos),
                             "pct": round(100 * pos / len(self.df), 1)})
        return pd.DataFrame(rows)

    # ── Class method: from split CSVs ────────────────────────────────────────

    @classmethod
    def from_mimic_dir(
        cls,
        data_dir: str | Path,
        split: str,
        **kwargs,
    ) -> "MIMICCXRDataset":
        """
        Load from a MIMIC-CXR directory with the standard processed layout:
            data_dir/
                metadata.csv   ← all splits with 'split' column
                images/        ← JPEG images
        """
        data_dir = Path(data_dir)
        csv_path = data_dir / "metadata.csv"
        image_root = data_dir / "images"
        return cls(csv_path, image_root=image_root, split=split, **kwargs)


# ── Synthetic demo dataset (no real data needed) ─────────────────────────────

class SyntheticCXRDataset(Dataset):
    """
    Generates synthetic (image, report) pairs for unit testing and demo.
    Images are random RGB tensors; reports are clinically plausible templates.
    """

    SYNTHETIC_REPORTS = [
        (
            "The heart size is normal. The mediastinum is not widened. "
            "The lungs are clear bilaterally. No pleural effusion or pneumothorax. "
            "No acute osseous abnormality.",
            "No acute cardiopulmonary process.",
        ),
        (
            "Heart size is mildly enlarged. There is mild pulmonary vascular "
            "congestion. Small bilateral pleural effusions. No pneumothorax.",
            "Cardiomegaly with mild congestive heart failure. Small bilateral pleural effusions.",
        ),
        (
            "There is increased opacity in the right lower lobe consistent with "
            "consolidation. The heart size is normal. No pleural effusion on the left. "
            "Small right pleural effusion cannot be excluded.",
            "Right lower lobe consolidation, possibly pneumonia. Correlate clinically.",
        ),
        (
            "Lung volumes are slightly low. There are basilar linear opacities "
            "consistent with atelectasis. The cardiac silhouette is within normal limits. "
            "No large pleural effusion or pneumothorax identified.",
            "Bibasilar atelectasis. No acute cardiopulmonary process.",
        ),
        (
            "There is a small left apical pneumothorax. The lungs are otherwise clear. "
            "Heart size is normal. No pleural effusion.",
            "Small left apical pneumothorax. Clinical correlation recommended.",
        ),
    ]

    def __init__(self, n_samples: int = 100, image_size: int = 448) -> None:
        self.n_samples = n_samples
        self.image_size = image_size

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> MIMICCXRSample:
        # Rotate through synthetic reports
        findings, impression = self.SYNTHETIC_REPORTS[idx % len(self.SYNTHETIC_REPORTS)]
        report = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"

        # Random grayscale-ish image (simulate CXR appearance)
        arr = np.random.randint(30, 200, (self.image_size, self.image_size), dtype=np.uint8)
        image = Image.fromarray(arr).convert("RGB")

        return MIMICCXRSample(
            study_id=f"synth_{idx:05d}",
            subject_id=f"p{idx % 10:05d}",
            image_path="",
            report=report,
            findings=findings,
            impression=impression,
            split="test",
            labels={"No Finding": float(idx % 2 == 0)},
            image=image,
        )
