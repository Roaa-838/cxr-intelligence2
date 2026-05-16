"""
src/rag/indexer.py
===================
RAGIndexer: high-level orchestrator that builds and manages both
ColPali and CLIP retrieval indexes from a MIMIC-CXR training set.

Usage:
    indexer = RAGIndexer(model="colpali")
    indexer.build(studies)          # list of {image, report, study_id}
    indexer.save("data/index")
    indexer.load("data/index")

    results = indexer.retrieve(image=pil_img, query="pleural effusion", top_k=5)
    context  = indexer.build_context(results)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

from PIL import Image

logger = logging.getLogger(__name__)


class RAGIndexer:
    """
    Unified indexer that wraps ColPali or CLIP retriever with a
    consistent interface for the RAG pipeline.

    Args:
        model:   "colpali" | "clip" | "biomedclip"
        mock:    Use mock models (no GPU required)
        top_k:   Default number of results to retrieve
    """

    def __init__(
        self,
        model: str = "colpali",
        mock: bool = False,
        top_k: int = 5,
    ) -> None:
        self.model_name = model
        self.mock = mock
        self.top_k = top_k
        self._retriever = None
        self._loaded = False

    # ── Loading ───────────────────────────────────────────────────────────────

    def _ensure_retriever(self) -> None:
        if self._retriever is not None:
            return
        if self.model_name == "colpali":
            from src.models.colpali_retriever import ColPaliRetriever
            self._retriever = ColPaliRetriever.load(mock=self.mock)
        else:
            from src.models.clip_retriever import CLIPRetriever
            clip_name = "biomedclip" if self.model_name in ("clip", "biomedclip") else self.model_name
            self._retriever = CLIPRetriever.load(model_name=clip_name, mock=self.mock)

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(
        self,
        studies: list[dict[str, Any]],
        batch_size: int = 4,
    ) -> "RAGIndexer":
        """
        Build the retrieval index from a list of studies.

        Args:
            studies:    list of dicts — required keys: "image" (PIL or path),
                        "report" (str), "study_id" (str)
            batch_size: embedding batch size (keep ≤4 for ColPali on GPU)

        Returns:
            self (chainable)
        """
        self._ensure_retriever()
        logger.info(
            "Building %s index over %d studies (batch_size=%d) …",
            self.model_name, len(studies), batch_size,
        )
        # Resolve any string paths to PIL Images
        for s in studies:
            img = s.get("image")
            if isinstance(img, str) or img is None:
                path = img or s.get("image_path", "")
                try:
                    s["image"] = Image.open(path).convert("RGB")
                except Exception:
                    s["image"] = Image.new("RGB", (448, 448), color=128)

        self._retriever.build_index(studies, batch_size=batch_size)
        self._loaded = True
        logger.info("Index built: %d entries.", self._retriever.index_size)
        return self

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, directory: Union[str, Path]) -> None:
        """Save index to directory."""
        self._ensure_retriever()
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        if self.model_name == "colpali":
            self._retriever.save_index(d / "colpali_index")
        else:
            self._retriever.save_index(d / "clip_index.pkl")
        logger.info("Index saved -> %s", d)

    def load(self, directory: Union[str, Path]) -> "RAGIndexer":
        """Load a previously saved index."""
        self._ensure_retriever()
        d = Path(directory)
        if self.model_name == "colpali":
            self._retriever.load_index(d / "colpali_index")
        else:
            self._retriever.load_index(d / "clip_index.pkl")
        self._loaded = True
        return self

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        image: Optional[Image.Image] = None,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most similar studies from the index.

        Args:
            image:  Query CXR image (PIL).
            query:  Query text string (clinical question or finding).
            top_k:  Override default K.

        Returns:
            list of dicts: study_id, score, report, rank
        """
        self._ensure_retriever()
        if not self._loaded:
            logger.warning(
                "Index not loaded. Call build() or load() first. "
                "Returning empty results."
            )
            return []
        k = top_k or self.top_k
        return self._retriever.retrieve(
            query_image=image,
            query_text=query,
            top_k=k,
        )

    def build_context(
        self,
        results: list[dict[str, Any]],
        max_chars: int = 2500,
    ) -> str:
        """Format retrieval results into a context string for the LLM."""
        self._ensure_retriever()
        if hasattr(self._retriever, "build_context_from_results"):
            return self._retriever.build_context_from_results(results, max_chars)
        # Fallback
        parts = []
        for r in results:
            report = r.get("report", "")[:500]
            parts.append(f"[Case {r.get('rank',1)} | score={r.get('score',0):.3f}]\n{report}")
        return "\n\n---\n\n".join(parts)[:max_chars]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._loaded

    @property
    def size(self) -> int:
        if self._retriever is None:
            return 0
        return getattr(self._retriever, "index_size", 0)

    def __repr__(self) -> str:
        return (
            f"RAGIndexer(model={self.model_name!r}, "
            f"loaded={self._loaded}, size={self.size})"
        )
