"""src/data/preprocessing.py — Chest X-ray image preprocessing pipeline."""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger(__name__)


class CXRPreprocessor:
    """
    End-to-end preprocessing for chest X-ray images.

    Clinical context
    ----------------
    CXRs are typically grayscale DICOM images converted to JPEG/PNG.
    They benefit from:
      - Histogram equalisation: distributes intensity more uniformly, enhancing
        subtle opacities (consolidation, effusion).
      - CLAHE (Contrast-Limited Adaptive Histogram Equalisation): local contrast
        enhancement without over-amplifying noise.
      - Normalisation: zero-mean, unit-variance per-channel; or ImageNet stats
        when using pretrained CNN backbones.

    Args:
        target_size:  (H, W) — MedGemma uses 448×448 or 896×896.
        apply_clahe:  Local contrast enhancement (recommended for pathologies).
        normalize:    Pixel normalisation mode: "imagenet" | "zero_one" | "zero_mean".
        augment:      Apply training-time augmentation (random flip, slight rotation).
    """

    # ImageNet statistics (used for CLIP and most CNN backbones)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    # SigLIP / MedGemma normalisation (uses 0–1 range, normalised differently)
    SIGLIP_MEAN = (0.5, 0.5, 0.5)
    SIGLIP_STD  = (0.5, 0.5, 0.5)

    def __init__(
        self,
        target_size: Tuple[int, int] = (448, 448),
        apply_clahe: bool = True,
        normalize: str = "siglip",   # "imagenet" | "siglip" | "zero_one" | "none"
        augment: bool = False,
    ) -> None:
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.normalize = normalize
        self.augment = augment
        self._build_transforms()

    def _build_transforms(self) -> None:
        """Construct torchvision transform pipeline."""
        ops: list = []

        # ── Resize ──────────────────────────────────────────────────────────
        ops.append(T.Resize(self.target_size, interpolation=T.InterpolationMode.LANCZOS))

        # ── Training augmentation ────────────────────────────────────────────
        if self.augment:
            ops += [
                T.RandomHorizontalFlip(p=0.5),          # CXR is left-right symmetric
                T.RandomRotation(degrees=5),              # Small rotation ±5°
                T.RandomAffine(degrees=0, translate=(0.02, 0.02)),  # Slight shift
            ]

        # ── To tensor ────────────────────────────────────────────────────────
        ops.append(T.ToTensor())

        # ── Normalisation ─────────────────────────────────────────────────────
        if self.normalize == "imagenet":
            ops.append(T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD))
        elif self.normalize == "siglip":
            ops.append(T.Normalize(mean=self.SIGLIP_MEAN, std=self.SIGLIP_STD))
        # "zero_one": already in [0,1] from ToTensor
        # "none": no normalisation

        self.transform = T.Compose(ops)

    # ── Public API ────────────────────────────────────────────────────────────

    def __call__(self, image: Union[Image.Image, str]) -> torch.Tensor:
        """
        Process a PIL image or path → normalised torch tensor.

        Returns:
            Float32 tensor of shape (3, H, W).
        """
        if isinstance(image, str):
            image = Image.open(image)
        image = self._to_rgb(image)
        if self.apply_clahe:
            image = self._apply_clahe(image)
        return self.transform(image)

    def process_pil(self, image: Image.Image) -> Image.Image:
        """Return a preprocessed PIL image (no tensor conversion)."""
        image = self._to_rgb(image)
        if self.apply_clahe:
            image = self._apply_clahe(image)
        image = image.resize(self.target_size, Image.LANCZOS)
        return image

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _to_rgb(image: Image.Image) -> Image.Image:
        """Convert any mode (L, RGBA, CMYK …) to RGB."""
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[3])
            return bg
        return image.convert("RGB")

    @staticmethod
    def _apply_clahe(image: Image.Image, clip_limit: float = 2.0) -> Image.Image:
        """
        Approximate CLAHE using PIL.
        For production, use cv2.createCLAHE for exact CLAHE.

        Clinical note: CLAHE is especially useful for revealing subtle
        interstitial opacities and pneumothorax lines.
        """
        try:
            import cv2  # type: ignore
            img_np = np.array(image.convert("L"))  # grayscale
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(img_np)
            return Image.fromarray(enhanced).convert("RGB")
        except ImportError:
            # Fallback: simple histogram equalisation with PIL
            gray = image.convert("L")
            equalized = ImageOps.equalize(gray)
            return equalized.convert("RGB")

    @staticmethod
    def denormalise(
        tensor: torch.Tensor,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std:  Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> torch.Tensor:
        """Reverse normalisation for visualisation."""
        m = torch.tensor(mean).view(3, 1, 1)
        s = torch.tensor(std).view(3, 1, 1)
        return tensor * s + m

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def for_medgemma(cls, augment: bool = False) -> "CXRPreprocessor":
        """Preprocessor configured for MedGemma-4B-it (448×448, SigLIP norm)."""
        return cls(target_size=(448, 448), apply_clahe=True,
                   normalize="siglip", augment=augment)

    @classmethod
    def for_clip(cls, augment: bool = False) -> "CXRPreprocessor":
        """Preprocessor configured for CLIP / BiomedCLIP (224×224, ImageNet norm)."""
        return cls(target_size=(224, 224), apply_clahe=False,
                   normalize="imagenet", augment=augment)

    @classmethod
    def for_colpali(cls) -> "CXRPreprocessor":
        """Preprocessor for ColPali (448×448, SigLIP norm, no augmentation)."""
        return cls(target_size=(448, 448), apply_clahe=True,
                   normalize="siglip", augment=False)


# ── Utility functions ─────────────────────────────────────────────────────────

def load_and_preprocess(
    image_path: str,
    preprocessor: Optional[CXRPreprocessor] = None,
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Load a CXR image and return both tensor and PIL for display.

    Returns:
        (tensor [3, H, W], PIL image)
    """
    if preprocessor is None:
        preprocessor = CXRPreprocessor.for_medgemma()

    img = Image.open(image_path)
    pil = preprocessor.process_pil(img)
    tensor = preprocessor(img)
    return tensor, pil


def validate_cxr_image(image: Image.Image) -> dict:
    """
    Basic quality checks for a chest X-ray image.

    Returns a dict with keys: valid (bool), warnings (list[str])
    """
    warnings: list[str] = []

    # Size check (very small images likely don't contain enough detail)
    w, h = image.size
    if w < 128 or h < 128:
        warnings.append(f"Image is very small ({w}×{h}px). Results may be poor.")

    # Aspect ratio (CXRs are roughly square to 4:3)
    ratio = w / h
    if ratio < 0.5 or ratio > 2.0:
        warnings.append(
            f"Unusual aspect ratio ({ratio:.2f}). Expected ~1.0–1.4 for PA CXR."
        )

    # Brightness (very dark or very bright suggest processing issues)
    arr = np.array(image.convert("L"), dtype=float)
    mean_brightness = arr.mean()
    if mean_brightness < 20:
        warnings.append("Image appears very dark. Possible exposure problem.")
    elif mean_brightness > 235:
        warnings.append("Image appears very bright. Possible over-exposure.")

    return {"valid": len(warnings) == 0, "warnings": warnings}
