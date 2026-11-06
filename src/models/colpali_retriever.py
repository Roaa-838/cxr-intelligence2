"""
src/models/colpali_retriever.py
================================
ColPali-v1.2 multimodal retriever for RAG-based CXR QA.

Paper: ColPali: Efficient Document Retrieval with Vision Language Models
       Faysse et al., ICLR 2025  (arXiv:2407.01449)

Architecture:
  - Base: PaliGemma-3B (SigLIP-So400m vision encoder + Gemma 2B language model)
  - Embedding: 1,024 patch embeddings × 128 dims per document image
  - Matching: Late interaction (ColBERT MaxSim):
      score(q, d) = Σ_i max_j  sim(q_i, d_j)
  
Clinical motivation for CXR:
  Each image patch corresponds to a spatial region of the X-ray (32×32 grid).
  A query about "left lower lobe opacity" will naturally attend to patches in
  the lower-left quadrant — unlike single-vector CLIP which loses spatial info.

Hardware:
  - ColPali inference: ~7 GB VRAM (bfloat16)
  - Indexing 10K CXRs: ~1.3 GB embeddings (float16), ~60 min on A100
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


class ColPaliRetriever(BaseRetriever):
    """
    Multimodal retriever using ColPali visual patch embeddings.

    Index CXR images as 1,024 patch vectors (32×32 grid, 128 dims each).
    At query time, embed the text question and compute MaxSim scores against
    all indexed patch embeddings — retrieval is grounded in visual regions.

    Usage:
        retriever = ColPaliRetriever.load()
        retriever.build_index(studies)
        results = retriever.retrieve(query_text="bilateral pleural effusion", top_k=5)
    """

    DEFAULT_MODEL_ID = "vidore/colpali-v1.2"
    EMBEDDING_DIM = 128
    N_PATCHES = 1024  # 32 × 32

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or self._detect_device()

        self._model = None
        self._processor = None

        # Index: (N, 1024, 128) array — one 32×32 patch grid per study
        self._patch_embeddings: Optional[np.ndarray] = None  # (N, 1024, 128)
        self._metadata: list[dict] = []
        self._is_indexed = False

    # ── Loading ────────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        mock: bool = False,
    ) -> "ColPaliRetriever":
        """
        Load ColPali model weights.

        Args:
            model_id: HuggingFace model ID (default: vidore/colpali-v1.2).
            device:   "cuda" | "cpu". Auto-detected if None.
            mock:     Return mock instance without loading weights.
        """
        if mock or os.environ.get("MOCK_MODELS", "").lower() in ("true", "1"):
            logger.info("ColPaliRetriever: loading in MOCK mode.")
            instance = cls(model_id=model_id, device="cpu")
            instance._mock = True
            return instance

        instance = cls(model_id=model_id, device=device)
        instance._mock = False
        instance._load_model()
        return instance

    def _load_model(self) -> None:
        """Load ColPali model and processor from HuggingFace."""
        try:
            from colpali_engine.models import ColPali, ColPaliProcessor
        except ImportError:
            raise ImportError(
                "colpali-engine not installed. Run: pip install colpali-engine"
            )

        logger.info("Loading ColPali: %s on %s …", self.model_id, self.device)

        self._processor = ColPaliProcessor.from_pretrained(self.model_id)
        self._model = ColPali.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self._model.eval()
        logger.info("ColPali loaded.")

    # ── Embedding ──────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def embed_image(self, image: Image.Image) -> torch.Tensor:
        """
        Embed a CXR image into ColPali patch embeddings.

        Returns:
            Float tensor of shape (1024, 128) — one vector per 32×32 patch.
            Note: This is NOT a single vector like CLIP. Use for MaxSim scoring.
        """
        if getattr(self, "_mock", False):
            return torch.randn(self.N_PATCHES, self.EMBEDDING_DIM)

        image = image.convert("RGB")
        batch = self._processor.process_images([image]).to(self._model.device)
        embeddings = self._model(**batch)  # (1, N_patches, 128)
        return embeddings[0].float().cpu()  # (1024, 128)

    @torch.inference_mode()
    def embed_text(self, text: str) -> torch.Tensor:
        """
        Embed a query text into ColPali query token embeddings.

        Returns:
            Float tensor of shape (N_query_tokens, 128).
        """
        if getattr(self, "_mock", False):
            n_tokens = min(len(text.split()) + 2, 32)
            return torch.randn(n_tokens, self.EMBEDDING_DIM)

        batch = self._processor.process_queries([text]).to(self._model.device)
        embeddings = self._model(**batch)  # (1, N_tokens, 128)
        return embeddings[0].float().cpu()  # (N_tokens, 128)

    @torch.inference_mode()
    def embed_images_batch(
        self,
        images: list[Image.Image],
        batch_size: int = 4,
    ) -> np.ndarray:
        """
        Batch-embed images. Returns (N, 1024, 128) numpy array (float16 to save memory).

        Note: ColPali is much heavier than CLIP per image — keep batch_size small.
        """
        if getattr(self, "_mock", False):
            return np.random.randn(len(images), self.N_PATCHES, self.EMBEDDING_DIM).astype(np.float16)

        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            batch_imgs = [img.convert("RGB") for img in images[i: i + batch_size]]
            batch = self._processor.process_images(batch_imgs).to(self._model.device)
            embs = self._model(**batch)  # (B, 1024, 128)
            all_embeddings.append(embs.float().cpu().numpy().astype(np.float16))
            logger.info("ColPali indexed %d / %d images …",
                        min(i + batch_size, len(images)), len(images))

        return np.concatenate(all_embeddings, axis=0)  # (N, 1024, 128)

    # ── MaxSim Scoring ─────────────────────────────────────────────────────────

    @staticmethod
    def maxsim_score(
        query_emb: torch.Tensor,   # (N_q, 128)
        doc_emb: torch.Tensor,     # (N_d, 128)
    ) -> float:
        """
        ColBERT late interaction MaxSim:
            score = Σ_i max_j  q_i · d_j

        Each query token finds its closest document patch and sums the scores.
        This allows spatial grounding: queries about specific regions attend
        to the corresponding spatial patches.
        """
        # (N_q, N_d) cosine similarity matrix
        sims = torch.einsum("qd,pd->qp",
                            F.normalize(query_emb.float(), dim=-1),
                            F.normalize(doc_emb.float(), dim=-1))
        return sims.max(dim=1).values.sum().item()  # sum of per-token maxima

    def score_query_against_index(
        self,
        query_emb: torch.Tensor,   # (N_q, 128)
        top_k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Score query against all indexed patch embeddings.

        Returns:
            (scores array of shape (N,), top_k indices)
        """
        N = self._patch_embeddings.shape[0]
        scores = np.zeros(N, dtype=np.float32)

        # Load in chunks to avoid OOM with large indices
        CHUNK = 256
        for start in range(0, N, CHUNK):
            chunk = torch.from_numpy(
                self._patch_embeddings[start: start + CHUNK].astype(np.float32)
            )  # (chunk, 1024, 128)
            for j, doc_emb in enumerate(chunk):
                scores[start + j] = self.maxsim_score(query_emb, doc_emb)

        top_idx = np.argsort(scores)[::-1][:top_k]
        return scores, top_idx

    # ── Indexing ───────────────────────────────────────────────────────────────

    def build_index(
        self,
        studies: list[dict[str, Any]],
        batch_size: int = 4,
    ) -> None:
        """
        Build ColPali patch embedding index from CXR studies.

        Args:
            studies:    list of dicts with keys: image (PIL/path), report, study_id
            batch_size: images per GPU batch (keep low for ColPali, ~4)
        """
        logger.info("Building ColPali index over %d studies …", len(studies))

        images: list[Image.Image] = []
        self._metadata = []

        from ..evaluation.validation import load_image
        for s in studies:
            img = load_image(s.get("image"), mock=getattr(self, "_mock", False))
            images.append(img)
            self._metadata.append({
                "study_id":   s.get("study_id", ""),
                "subject_id": s.get("subject_id", ""),
                "split": s.get("split", ""),
                "data_mode": s.get("data_mode", "unverified"),
                "model_mode": "mock" if getattr(self, "_mock", False) else "real",
                "report":     s.get("report", ""),
                "findings":   s.get("findings", ""),
                "impression": s.get("impression", ""),
                "image_path": s.get("image_path", ""),
            })

        self._patch_embeddings = self.embed_images_batch(images, batch_size=batch_size)
        self._is_indexed = True
        shape = self._patch_embeddings.shape
        mem_mb = self._patch_embeddings.nbytes / 1e6
        logger.info(
            "ColPali index built: %d studies × %d patches × %d dims (%.1f MB)",
            shape[0], shape[1], shape[2], mem_mb
        )

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_image: Optional[Image.Image] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top_k studies using MaxSim late interaction.

        Strategy:
          - If query_text is given: embed text → MaxSim against image patches
          - If query_image is given: embed image → MaxSim against index (image-to-image)
          - If both: average their MaxSim scores

        Returns:
            list of dicts: study_id, score, report, image_path, rank
        """
        if getattr(self, "_mock", False):
            from .base import MockRetriever
            return MockRetriever().retrieve(query_image, query_text, top_k)

        if not self._is_indexed:
            raise RuntimeError(
                "Index not built. Call build_index() or load_index() first."
            )

        scores_combined = np.zeros(len(self._metadata), dtype=np.float32)
        n_queries = 0

        if query_text is not None:
            txt_emb = self.embed_text(query_text)   # (N_q, 128)
            scores_txt, _ = self.score_query_against_index(txt_emb, top_k=len(self._metadata))
            scores_combined += scores_txt
            n_queries += 1

        if query_image is not None:
            img_emb = self.embed_image(query_image)   # (1024, 128)
            scores_img, _ = self.score_query_against_index(img_emb, top_k=len(self._metadata))
            scores_combined += scores_img
            n_queries += 1

        if n_queries > 1:
            scores_combined /= n_queries

        top_indices = np.argsort(scores_combined)[::-1][:top_k]
        results: list[dict] = []
        for rank, idx in enumerate(top_indices):
            meta = self._metadata[idx].copy()
            meta["score"] = float(scores_combined[idx])
            meta["rank"] = rank + 1
            results.append(meta)

        return results

    def build_context_from_results(
        self,
        results: list[dict[str, Any]],
        max_context_length: int = 2000,
    ) -> str:
        """Format retrieved results into LLM context string."""
        parts: list[str] = []
        for r in results:
            report = r.get("report", r.get("findings", ""))[:500]
            parts.append(
                f"[Case {r['rank']} | ColPali score={r['score']:.3f}]\n{report}"
            )
        context = "\n\n---\n\n".join(parts)
        return context[:max_context_length]

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_index(self, path: str | Path) -> None:
        """Save patch embeddings + metadata to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.save(str(path) + ".npy", self._patch_embeddings)
        with open(str(path) + ".meta.pkl", "wb") as f:
            pickle.dump(self._metadata, f)

        logger.info("ColPali index saved -> %s (.npy + .meta.pkl)", path)

    def load_index(self, path: str | Path) -> None:
        """Load a previously saved ColPali index."""
        path = Path(path)
        self._patch_embeddings = np.load(str(path) + ".npy")
        with open(str(path) + ".meta.pkl", "rb") as f:
            self._metadata = pickle.load(f)
        self._is_indexed = True
        logger.info("ColPali index loaded <- %s (%d entries)", path, len(self._metadata))

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return f"ColPaliRetriever({self.model_id.split('/')[-1]})"

    @property
    def is_indexed(self) -> bool:
        return self._is_indexed

    @property
    def index_size(self) -> int:
        return len(self._metadata) if self._is_indexed else 0

    @staticmethod
    def _detect_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def __repr__(self) -> str:
        return (
            f"ColPaliRetriever(model={self.model_id!r}, "
            f"device={self.device!r}, "
            f"indexed={self._is_indexed}, "
            f"n_entries={self.index_size})"
        )
