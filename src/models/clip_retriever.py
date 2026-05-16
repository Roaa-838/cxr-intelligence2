"""
src/models/clip_retriever.py
============================
CLIP and BiomedCLIP retrieval models for image-text similarity search.

Models supported:
  - BiomedCLIP (Microsoft): trained on 15M biomedical image-text pairs from PMC
  - CLIP ViT-L/14 (OpenAI): baseline general-purpose CLIP
  - Custom model_id via open_clip

Role in the system:
  - Embed CXR images and report text into a shared vector space
  - Retrieve similar training studies given a query image (for RAG)
  - Provide alternative retrieval comparison against ColPali
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base import BaseRetriever

logger = logging.getLogger(__name__)


class CLIPRetriever(BaseRetriever):
    """
    Image-text retrieval using CLIP or BiomedCLIP embeddings.

    Indexing workflow:
        retriever = CLIPRetriever.load("biomedclip")
        retriever.build_index(study_list)   # list of {image, report, study_id}
        retriever.save_index("data/index/clip_index.pkl")

    Retrieval workflow:
        retriever = CLIPRetriever.load("biomedclip")
        retriever.load_index("data/index/clip_index.pkl")
        results = retriever.retrieve(query_image=pil_img, query_text="pleural effusion", top_k=5)
    """

    SUPPORTED_MODELS = {
        "biomedclip": {
            "model_id": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            "pretrained": None,
            "embedding_dim": 512,
            "image_size": 224,
        },
        "clip-vit-l-14": {
            "model_id": "ViT-L-14",
            "pretrained": "openai",
            "embedding_dim": 768,
            "image_size": 224,
        },
        "clip-vit-b-32": {
            "model_id": "ViT-B-32",
            "pretrained": "openai",
            "embedding_dim": 512,
            "image_size": 224,
        },
    }

    def __init__(
        self,
        model_name: str = "biomedclip",
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self.device = device or self._detect_device()
        self._model = None
        self._preprocess = None
        self._tokenizer = None

        # Index storage
        self._image_embeddings: Optional[np.ndarray] = None  # (N, D)
        self._metadata: list[dict] = []
        self._is_indexed = False

    # ── Loading ────────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        model_name: str = "biomedclip",
        device: Optional[str] = None,
        mock: bool = False,
    ) -> "CLIPRetriever":
        """
        Load CLIP / BiomedCLIP model.

        Args:
            model_name: "biomedclip" | "clip-vit-l-14" | "clip-vit-b-32"
            device:     "cuda" | "cpu" | None (auto-detect)
            mock:       Return unfilled instance (for testing without GPU)
        """
        if mock or os.environ.get("MOCK_MODELS", "").lower() in ("true", "1"):
            logger.info("CLIPRetriever: loading in MOCK mode.")
            instance = cls(model_name=model_name, device="cpu")
            instance._mock = True
            return instance

        instance = cls(model_name=model_name, device=device)
        instance._mock = False
        instance._load_model()
        return instance

    def _load_model(self) -> None:
        """Load open_clip model and preprocessor."""
        import open_clip

        cfg = self.SUPPORTED_MODELS.get(self._model_name)
        if cfg is None:
            raise ValueError(
                f"Unknown model '{self._model_name}'. "
                f"Supported: {list(self.SUPPORTED_MODELS)}"
            )

        model_id = cfg["model_id"]
        pretrained = cfg.get("pretrained")

        logger.info("Loading CLIP model: %s (pretrained=%s) …", model_id, pretrained)

        if model_id.startswith("hf-hub:"):
            # BiomedCLIP — load via HF hub
            self._model, self._preprocess = open_clip.create_model_from_pretrained(
                model_id
            )
            self._tokenizer = open_clip.get_tokenizer(model_id)
        else:
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_id, pretrained=pretrained
            )
            self._tokenizer = open_clip.get_tokenizer(model_id)

        self._model = self._model.to(self.device)
        self._model.eval()
        logger.info("CLIP model loaded on %s.", self.device)

    # ── Embedding ──────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def embed_image(self, image: Image.Image) -> torch.Tensor:
        """
        Embed a PIL image into the CLIP visual feature space.

        Returns:
            L2-normalised float32 tensor of shape (embedding_dim,)
        """
        if getattr(self, "_mock", False):
            dim = self.SUPPORTED_MODELS.get(self._model_name, {}).get("embedding_dim", 512)
            return F.normalize(torch.randn(dim), dim=0)

        image = image.convert("RGB")
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        features = self._model.encode_image(tensor)
        return F.normalize(features.squeeze(0).float(), dim=0).cpu()

    @torch.inference_mode()
    def embed_text(self, text: str) -> torch.Tensor:
        """
        Embed a text string into the CLIP text feature space.

        Returns:
            L2-normalised float32 tensor of shape (embedding_dim,)
        """
        if getattr(self, "_mock", False):
            dim = self.SUPPORTED_MODELS.get(self._model_name, {}).get("embedding_dim", 512)
            return F.normalize(torch.randn(dim), dim=0)

        tokens = self._tokenizer([text]).to(self.device)
        features = self._model.encode_text(tokens)
        return F.normalize(features.squeeze(0).float(), dim=0).cpu()

    @torch.inference_mode()
    def embed_images_batch(
        self,
        images: list[Image.Image],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Batch-embed a list of PIL images efficiently.

        Returns:
            numpy array of shape (N, embedding_dim), L2-normalised.
        """
        if getattr(self, "_mock", False):
            dim = self.SUPPORTED_MODELS.get(self._model_name, {}).get("embedding_dim", 512)
            return np.random.randn(len(images), dim).astype(np.float32)

        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            batch = images[i: i + batch_size]
            tensors = torch.stack([
                self._preprocess(img.convert("RGB")) for img in batch
            ]).to(self.device)
            feats = self._model.encode_image(tensors)
            feats = F.normalize(feats.float(), dim=-1).cpu().numpy()
            all_embeddings.append(feats)
            if (i // batch_size) % 10 == 0:
                logger.info("Embedded %d / %d images …", min(i + batch_size, len(images)), len(images))

        return np.concatenate(all_embeddings, axis=0)

    # ── Indexing ───────────────────────────────────────────────────────────────

    def build_index(
        self,
        studies: list[dict[str, Any]],
        batch_size: int = 32,
    ) -> None:
        """
        Build an in-memory retrieval index from a list of training studies.

        Args:
            studies: list of dicts with keys:
                       - "image"      (PIL Image or path string)
                       - "report"     (str)
                       - "study_id"   (str)
                       - "findings"   (str, optional)
                       - "impression" (str, optional)
            batch_size: Embedding batch size.
        """
        logger.info("Building CLIP index over %d studies …", len(studies))

        images: list[Image.Image] = []
        self._metadata = []

        for s in studies:
            img = s["image"]
            if isinstance(img, str):
                try:
                    img = Image.open(img).convert("RGB")
                except Exception:
                    logger.warning("Could not load image: %s", img)
                    img = Image.new("RGB", (224, 224), color=128)
            images.append(img)
            self._metadata.append({
                "study_id":   s.get("study_id", ""),
                "report":     s.get("report", ""),
                "findings":   s.get("findings", ""),
                "impression": s.get("impression", ""),
                "image_path": s.get("image_path", ""),
            })

        self._image_embeddings = self.embed_images_batch(images, batch_size=batch_size)
        self._is_indexed = True
        logger.info("Index built: %d vectors × %d dims.",
                    self._image_embeddings.shape[0], self._image_embeddings.shape[1])

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_image: Optional[Image.Image] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
        alpha: float = 0.7,   # weight for image similarity vs text
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top_k most similar studies.

        Hybrid scoring:
            score = alpha * image_sim + (1 - alpha) * text_sim
        Set alpha=1.0 for pure image retrieval, 0.0 for pure text retrieval.

        Args:
            query_image: PIL Image to use as visual query.
            query_text:  Text string for text-based retrieval.
            top_k:       Number of results to return.
            alpha:       Image-text interpolation weight.

        Returns:
            list of dicts: study_id, score, report, image_path, rank
        """
        if getattr(self, "_mock", False):
            from .base import MockRetriever
            return MockRetriever().retrieve(query_image, query_text, top_k)

        if not self._is_indexed:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")

        scores = np.zeros(len(self._metadata), dtype=np.float32)

        if query_image is not None:
            img_emb = self.embed_image(query_image).numpy()      # (D,)
            img_scores = self._image_embeddings @ img_emb         # (N,)
            scores += alpha * img_scores

        if query_text is not None:
            txt_emb = self.embed_text(query_text).numpy()         # (D,)
            txt_scores = self._image_embeddings @ txt_emb         # (N,) using image index
            scores += (1.0 - alpha) * txt_scores

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: list[dict] = []
        for rank, idx in enumerate(top_indices):
            meta = self._metadata[idx].copy()
            meta["score"] = float(scores[idx])
            meta["rank"]  = rank + 1
            results.append(meta)

        return results

    def build_context_from_results(
        self,
        results: list[dict[str, Any]],
        max_context_length: int = 2000,
    ) -> str:
        """
        Format retrieved results as a context string for the LLM prompt.

        Args:
            results:            Output of retrieve().
            max_context_length: Truncate context to this many characters.

        Returns:
            Formatted multi-study context string.
        """
        parts: list[str] = []
        for r in results:
            report = r.get("report", r.get("findings", ""))
            if len(report) > 500:
                report = report[:500] + "..."
            parts.append(
                f"[Case {r['rank']} | similarity={r['score']:.3f}]\n{report}"
            )

        context = "\n\n---\n\n".join(parts)
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n[Context truncated]"
        return context

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_index(self, path: str | Path) -> None:
        """Save index to disk (numpy array + metadata pickle)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_name": self._model_name,
            "embeddings": self._image_embeddings,
            "metadata":   self._metadata,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Saved CLIP index -> %s (%d entries)", path, len(self._metadata))

    def load_index(self, path: str | Path) -> None:
        """Load a previously saved index."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._image_embeddings = data["embeddings"]
        self._metadata = data["metadata"]
        self._is_indexed = True
        logger.info("Loaded CLIP index <- %s (%d entries)", path, len(self._metadata))

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return f"CLIPRetriever({self._model_name})"

    @property
    def is_indexed(self) -> bool:
        return self._is_indexed

    @property
    def index_size(self) -> int:
        return len(self._metadata) if self._is_indexed else 0

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def __repr__(self) -> str:
        return (
            f"CLIPRetriever(model={self._model_name!r}, "
            f"device={self.device!r}, "
            f"indexed={self._is_indexed}, "
            f"n_entries={self.index_size})"
        )
