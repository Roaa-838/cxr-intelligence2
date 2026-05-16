"""
src/rag/store.py
=================
VectorStore: unified interface over Qdrant and FAISS vector databases.
Used when you want persistent, server-backed storage vs the in-memory
numpy arrays used by the retriever classes.

Recommended usage:
  - Development / small index (≤50K vectors): use CLIPRetriever/ColPaliRetriever
    directly (numpy-backed, zero infra).
  - Production / large index (>50K vectors):  use VectorStore (Qdrant).

Qdrant modes:
  - In-memory:   VectorStore(backend="qdrant", path=":memory:")
  - On-disk:     VectorStore(backend="qdrant", path="./data/qdrant")
  - Remote:      VectorStore(backend="qdrant", url="http://localhost:6333")
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Thin wrapper around Qdrant (primary) or FAISS (fallback) for
    storing and querying study embeddings.

    Args:
        backend:         "qdrant" | "faiss"
        collection_name: Qdrant collection name
        embedding_dim:   Vector dimensionality (512 for CLIP, 128 for ColPali mean)
        path:            Qdrant on-disk path or ":memory:"
        url:             Qdrant server URL (overrides path if given)
    """

    def __init__(
        self,
        backend: str = "qdrant",
        collection_name: str = "cxr_studies",
        embedding_dim: int = 512,
        path: str = ":memory:",
        url: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.path = path
        self.url = url
        self._store = None
        self._faiss_index = None
        self._faiss_meta: list[dict] = []

    # ── Init ──────────────────────────────────────────────────────────────────

    def connect(self) -> "VectorStore":
        """Initialise the backend connection and create collection if needed."""
        if self.backend == "qdrant":
            self._connect_qdrant()
        elif self.backend == "faiss":
            self._connect_faiss()
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")
        return self

    def _connect_qdrant(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            if self.url:
                self._store = QdrantClient(url=self.url)
                logger.info("Connected to Qdrant server: %s", self.url)
            elif self.path == ":memory:":
                self._store = QdrantClient(":memory:")
                logger.info("Using in-memory Qdrant.")
            else:
                self._store = QdrantClient(path=self.path)
                logger.info("Using on-disk Qdrant: %s", self.path)

            # Create collection if it doesn't exist
            existing = [c.name for c in self._store.get_collections().collections]
            if self.collection_name not in existing:
                self._store.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "Created Qdrant collection '%s' (dim=%d).",
                    self.collection_name, self.embedding_dim,
                )
        except ImportError:
            logger.warning("qdrant-client not installed — falling back to FAISS.")
            self.backend = "faiss"
            self._connect_faiss()

    def _connect_faiss(self) -> None:
        try:
            import faiss
            self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product
            logger.info("FAISS index created (dim=%d).", self.embedding_dim)
        except ImportError:
            logger.error(
                "Neither qdrant-client nor faiss-cpu is installed. "
                "Run: pip install qdrant-client  OR  pip install faiss-cpu"
            )
            raise

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(
        self,
        study_id: str,
        embedding: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        """
        Insert or update one study embedding.

        Args:
            study_id:  Unique study identifier.
            embedding: 1-D float32 numpy array of shape (embedding_dim,).
            metadata:  Arbitrary payload stored alongside the vector.
        """
        embedding = embedding.astype(np.float32)
        if self.backend == "qdrant":
            self._qdrant_upsert(study_id, embedding, metadata)
        else:
            self._faiss_upsert(embedding, {**metadata, "study_id": study_id})

    def upsert_batch(
        self,
        study_ids: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]],
        batch_size: int = 256,
    ) -> None:
        """Batch upsert for efficiency."""
        n = len(study_ids)
        for i in range(0, n, batch_size):
            batch_ids  = study_ids[i: i + batch_size]
            batch_embs = embeddings[i: i + batch_size]
            batch_meta = metadatas[i: i + batch_size]
            for sid, emb, meta in zip(batch_ids, batch_embs, batch_meta):
                self.upsert(sid, emb, meta)
            logger.info("Upserted %d / %d vectors …", min(i + batch_size, n), n)

    def _qdrant_upsert(self, study_id, embedding, metadata):
        from qdrant_client.models import PointStruct
        point_id = abs(hash(study_id)) % (2 ** 53)
        self._store.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={"study_id": study_id, **metadata},
            )],
        )

    def _faiss_upsert(self, embedding, metadata):
        import faiss
        vec = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        self._faiss_index.add(vec)
        self._faiss_meta.append(metadata)

    # ── Query ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Find the top_k most similar vectors to the query.

        Args:
            query_embedding: 1-D float32 array of shape (embedding_dim,).
            top_k:           Number of results to return.
            score_threshold: Minimum similarity score (0–1 for cosine).

        Returns:
            list of dicts with keys: study_id, score, + all metadata fields.
        """
        query = query_embedding.astype(np.float32)
        if self.backend == "qdrant":
            return self._qdrant_search(query, top_k, score_threshold)
        return self._faiss_search(query, top_k, score_threshold)

    def _qdrant_search(self, query, top_k, threshold):
        try:
            response = self._store.query_points(
                collection_name=self.collection_name,
                query=query.tolist(),
                limit=top_k,
                score_threshold=threshold if threshold > 0 else None,
            )
            hits = response.points
        except AttributeError:
            hits = self._store.search(
                collection_name=self.collection_name,
                query_vector=query.tolist(),
                limit=top_k,
                score_threshold=threshold if threshold > 0 else None,
            )
        results = []
        for h in hits:
            r = dict(h.payload)
            r["score"] = h.score
            results.append(r)
        return results

    def _faiss_search(self, query, top_k, threshold):
        import faiss
        vec = query.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        scores, indices = self._faiss_index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < threshold:
                continue
            r = dict(self._faiss_meta[idx])
            r["score"] = float(score)
            results.append(r)
        return results

    # ── Info ──────────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        """Number of vectors in the store."""
        if self.backend == "qdrant" and self._store:
            return self._store.get_collection(self.collection_name).points_count
        if self.backend == "faiss" and self._faiss_index:
            return self._faiss_index.ntotal
        return 0

    def __repr__(self) -> str:
        return (
            f"VectorStore(backend={self.backend!r}, "
            f"collection={self.collection_name!r}, "
            f"dim={self.embedding_dim}, count={self.count})"
        )
