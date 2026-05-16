"""
src/pipelines/rag_qa.py
========================
Mode 2 — RAG-Based Clinical QA Pipeline

Flow:
  1. Receive (CXR image, clinical question)
  2. Embed query with ColPali or CLIP retriever
  3. Retrieve top-K similar studies from indexed knowledge base
  4. Build grounded context from retrieved reports
  5. Generate answer with MedGemma using image + context
  6. Return answer with provenance (retrieved study IDs, scores)

Supports two retriever backends:
  - ColPali:    multimodal patch-level embedding (visual + spatial)
  - CLIP/BiomedCLIP: single-vector image+text embedding
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Union

from PIL import Image

logger = logging.getLogger(__name__)


class RAGQAPipeline:
    """
    Retrieval-Augmented Generation pipeline for CXR clinical QA.

    Args:
        retriever:  "colpali" | "clip" | "mock"
        generator:  "medgemma" (only VLM supported; extend as needed)
        cfg:        Config dict from load_config().
        mock:       Use mock models (no GPU required).

    Example:
        pipeline = RAGQAPipeline(retriever="colpali", mock=True).load()
        result = pipeline.run(image=pil_image, question="Is there a pneumothorax?")
        print(result["answer"])
        print(result["retrieved_studies"])
    """

    def __init__(
        self,
        retriever: str = "colpali",
        generator: str = "medgemma",
        cfg: Optional[dict] = None,
        mock: bool = False,
    ) -> None:
        self.retriever_name = retriever
        self.generator_name = generator
        self.cfg = cfg or {}
        self.mock = mock or (retriever == "mock")

        self._retriever = None
        self._generator = None
        self._is_ready = False

    # ── Lazy loading ──────────────────────────────────────────────────────────

    def load(self) -> "RAGQAPipeline":
        """Load retriever and generator. Call before run()."""
        logger.info(
            "Loading RAGQAPipeline (retriever=%s, generator=%s) …",
            self.retriever_name, self.generator_name,
        )
        if self.mock:
            self._load_mock()
        else:
            self._load_retriever()
            self._load_generator()

        self._is_ready = True
        logger.info("RAGQAPipeline ready.")
        return self

    def _load_mock(self) -> None:
        from ..models.base import MockRetriever, MockVLM
        self._retriever = MockRetriever()
        self._generator = MockVLM()

    def _load_retriever(self) -> None:
        import os
        from pathlib import Path

        index_base = Path(self.cfg.get("rag", {}).get("index_path", "data/index"))

        if self.retriever_name == "colpali":
            from ..models.colpali_retriever import ColPaliRetriever
            self._retriever = ColPaliRetriever.load()
            index_file = index_base / "colpali_index"
            if (Path(str(index_file) + ".npy")).exists():
                logger.info("Loading ColPali index from %s", index_file)
                self._retriever.load_index(index_file)
            else:
                logger.warning(
                    "ColPali index not found at %s. "
                    "Run: python scripts/build_index.py --model colpali", index_file
                )

        elif self.retriever_name in ("clip", "biomedclip"):
            from ..models.clip_retriever import CLIPRetriever
            model_name = (
                "biomedclip" if self.retriever_name in ("clip", "biomedclip")
                else "clip-vit-l-14"
            )
            self._retriever = CLIPRetriever.load(model_name=model_name)
            index_file = index_base / "clip_index.pkl"
            if index_file.exists():
                logger.info("Loading CLIP index from %s", index_file)
                self._retriever.load_index(index_file)
            else:
                logger.warning("CLIP index not found at %s.", index_file)
        else:
            raise ValueError(
                f"Unknown retriever '{self.retriever_name}'. "
                "Choose: 'colpali' | 'clip'"
            )

    def _load_generator(self) -> None:
        from ..models.medgemma import MedGemmaModel
        model_cfg = self.cfg.get("models", {}).get("medgemma", {})
        self._generator = MedGemmaModel.load(
            use_4bit=model_cfg.get("use_4bit", False),
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def run(
        self,
        image: Union[Image.Image, str],
        question: str,
        top_k: int = 5,
        qa_prompt: str = "rag",
        max_new_tokens: int = 256,
        return_context: bool = True,
    ) -> dict[str, Any]:
        """
        Answer a clinical question about a CXR using RAG.

        Args:
            image:          PIL Image or path.
            question:       Clinical question (free text).
            top_k:          Number of similar studies to retrieve.
            qa_prompt:      Prompt template name: "rag" | "zero_shot" | "structured"
            max_new_tokens: Max generated tokens.
            return_context: Include retrieved context in output dict.

        Returns:
            dict with keys:
              "question":         str
              "answer":           str
              "retrieved_studies": list[str] — study IDs
              "retrieval_scores": list[float]
              "context":          str (if return_context=True)
              "retriever":        str
              "generator":        str
              "latency_s":        float
        """
        if not self._is_ready:
            self.load()

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        t0 = time.time()

        # ── Step 1: Retrieve ──────────────────────────────────────────────────
        retrieved = self._retriever.retrieve(
            query_image=image,
            query_text=question,
            top_k=top_k,
        )

        # ── Step 2: Build context ─────────────────────────────────────────────
        context = self._build_context(retrieved)

        # ── Step 3: Generate answer ───────────────────────────────────────────
        answer = self._generator.answer_question(
            image=image,
            question=question,
            context=context,
            max_new_tokens=max_new_tokens,
        )

        latency = round(time.time() - t0, 2)

        result: dict[str, Any] = {
            "question":          question,
            "answer":            answer,
            "retrieved_studies": [r.get("study_id", "") for r in retrieved],
            "retrieval_scores":  [r.get("score", 0.0) for r in retrieved],
            "retriever":         self.retriever_name,
            "generator":         self.generator_name,
            "latency_s":         latency,
        }
        if return_context:
            result["context"] = context
            result["retrieved_details"] = retrieved

        return result

    def run_batch(
        self,
        qa_pairs: list[dict],
        top_k: int = 5,
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Run QA on a batch of (image, question) pairs.

        Args:
            qa_pairs: list of dicts with keys:
                        "image"    (PIL or path)
                        "question" (str)
                        "answer"   (str, reference — optional)
                        "study_id" (str — optional)
            top_k:    Retrieval K.
            show_progress: Log progress every 10 samples.

        Returns:
            list of result dicts including "reference_answer" where provided.
        """
        if not self._is_ready:
            self.load()

        results: list[dict] = []
        n = len(qa_pairs)

        for i, pair in enumerate(qa_pairs):
            try:
                img = pair.get("image")
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")

                result = self.run(
                    image=img,
                    question=pair["question"],
                    top_k=top_k,
                    return_context=False,
                )
                result["study_id"]        = pair.get("study_id", f"sample_{i}")
                result["reference_answer"] = pair.get("answer", "")
                results.append(result)

            except Exception as e:
                logger.warning("Failed on QA sample %d: %s", i, e)
                results.append({
                    "study_id":         pair.get("study_id", f"sample_{i}"),
                    "question":         pair.get("question", ""),
                    "answer":           f"ERROR: {e}",
                    "reference_answer": pair.get("answer", ""),
                    "error":            str(e),
                    "latency_s":        0.0,
                })

            if show_progress and (i + 1) % 10 == 0:
                logger.info("  QA progress: %d / %d", i + 1, n)

        return results

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_context(
        retrieved: list[dict],
        max_chars_per_report: int = 600,
        max_total_chars: int = 2500,
    ) -> str:
        """Format retrieved studies into a prompt context block."""
        parts: list[str] = []
        total_chars = 0
        for r in retrieved:
            report = r.get("report", r.get("findings", ""))
            if len(report) > max_chars_per_report:
                report = report[:max_chars_per_report] + "…"
            score = r.get("score", 0.0)
            sid   = r.get("study_id", "?")
            block = f"[Study {sid} | similarity={score:.3f}]\n{report}"
            if total_chars + len(block) > max_total_chars:
                break
            parts.append(block)
            total_chars += len(block)
        return "\n\n---\n\n".join(parts)

    def set_index(self, studies: list[dict], batch_size: int = 4) -> None:
        """
        Build retrieval index from provided studies (in-memory convenience method).

        Args:
            studies:    list of dicts with keys: image, report, study_id
            batch_size: embedding batch size
        """
        if not self._is_ready:
            self.load()
        logger.info("Building %s index over %d studies …",
                    self.retriever_name, len(studies))
        self._retriever.build_index(studies, batch_size=batch_size)

    @property
    def is_indexed(self) -> bool:
        return self._retriever is not None and self._retriever.is_indexed
