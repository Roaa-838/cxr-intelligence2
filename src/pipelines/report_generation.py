"""
src/pipelines/report_generation.py
====================================
Mode 1 — Report Generation Pipeline

Four sub-modes, all sharing a single .generate() interface:

  A. MedGemma zero-shot   — image → MedGemma-4B-it → FINDINGS + IMPRESSION
  B. MedGemma fine-tuned  — image → QLoRA checkpoint → FINDINGS + IMPRESSION
  C. CLIP-RAG             — image → BiomedCLIP retrieve top-K → MedGemma generate
  D. ColPali-RAG          — image → ColPali retrieve top-K   → MedGemma generate

Quick start
-----------
    pipeline = ReportGenerationPipeline(model="medgemma").load()
    report   = pipeline.generate(pil_image)
    print(report.findings)
    print(report.impression)

    # Batch evaluation
    results = pipeline.generate_batch(sample_list, show_progress=True)

    # Side-by-side comparison
    ReportGenerationPipeline.compare(report_a, report_b, reference="...")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReportResult:
    """
    Structured output of one report generation call.

    Attributes
    ----------
    findings:       FINDINGS section text.
    impression:     IMPRESSION section text.
    raw:            Full raw model output (before section parsing).
    model:          Model name used.
    latency_s:      Wall-clock inference time in seconds.
    study_id:       Caller-supplied study identifier (optional).
    reference:      Ground-truth report for evaluation (optional).
    context:        Retrieved context used in RAG mode (optional).
    retrieved_ids:  Study IDs of retrieved cases (RAG mode, optional).
    error:          Error message if inference failed (optional).
    """
    findings:       str       = ""
    impression:     str       = ""
    raw:            str       = ""
    model:          str       = ""
    latency_s:      float     = 0.0
    study_id:       str       = ""
    reference:      str       = ""
    context:        str       = ""
    retrieved_ids:  list[str] = field(default_factory=list)
    error:          str       = ""

    @property
    def succeeded(self) -> bool:
        return not bool(self.error)

    @property
    def full_report(self) -> str:
        """Canonical formatted report string."""
        if self.findings and self.impression:
            return f"FINDINGS:\n{self.findings}\n\nIMPRESSION:\n{self.impression}"
        return self.raw

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings":      self.findings,
            "impression":    self.impression,
            "raw":           self.raw,
            "model":         self.model,
            "latency_s":     self.latency_s,
            "study_id":      self.study_id,
            "reference":     self.reference,
            "context":       self.context,
            "retrieved_ids": self.retrieved_ids,
            "error":         self.error,
        }

    def __str__(self) -> str:
        return self.full_report


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ReportGenerationPipeline:
    """
    Unified chest X-ray report generation pipeline.

    Supported modes
    ---------------
    ``medgemma``     MedGemma-4B-it zero-shot (default)
    ``medgemma-ft``  Fine-tuned MedGemma checkpoint (QLoRA)
    ``clip-rag``     BiomedCLIP retrieval + MedGemma generation
    ``colpali-rag``  ColPali retrieval   + MedGemma generation
    ``mock``         Deterministic mock output  (no GPU required)
    """

    SUPPORTED_MODELS = frozenset({
        "medgemma", "medgemma-ft",
        "clip-rag", "colpali-rag",
        "mock",
    })

    def __init__(
        self,
        model: str = "medgemma",
        cfg:   Optional[dict] = None,
        mock:  bool = False,
    ) -> None:
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model '{model}'. "
                f"Choose from: {sorted(self.SUPPORTED_MODELS)}"
            )
        self.model_name = model
        self.cfg        = cfg or {}
        self.mock       = mock or (model == "mock")

        self._vlm       = None
        self._retriever = None
        self._is_ready  = False

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> "ReportGenerationPipeline":
        """
        Load model weights into memory.
        Returns self so you can chain:  pipeline = Pipeline(...).load()
        """
        logger.info("Loading pipeline  model=%s  mock=%s", self.model_name, self.mock)
        loader = {
            "mock":        self._load_mock,
            "medgemma":    self._load_medgemma,
            "medgemma-ft": self._load_medgemma,
            "clip-rag":    self._load_clip_rag,
            "colpali-rag": self._load_colpali_rag,
        }[self.model_name if not self.mock else "mock"]
        loader()
        self._is_ready = True
        logger.info("Pipeline ready  model=%s", self.model_name)
        return self

    def _load_mock(self) -> None:
        from ..models.base import MockVLM, MockRetriever
        self._vlm       = MockVLM()
        self._retriever = MockRetriever()

    def _load_medgemma(self) -> None:
        from ..models.medgemma import MedGemmaModel
        model_cfg = self.cfg.get("models", {}).get("medgemma", {})
        model_id  = model_cfg.get("model_id", "google/medgemma-4b-it")
        if self.model_name == "medgemma-ft":
            ft_dir = self.cfg.get("finetune", {}).get(
                "output_dir", "checkpoints/medgemma-cxr-ft/final"
            )
            if Path(ft_dir).exists():
                model_id = ft_dir
                logger.info("Fine-tuned checkpoint: %s", ft_dir)
            else:
                logger.warning(
                    "Fine-tuned checkpoint not found at '%s'. "
                    "Falling back to base MedGemma. "
                    "Run: python scripts/finetune_medgemma.py", ft_dir
                )
        self._vlm = MedGemmaModel.load(
            model_id=model_id,
            use_4bit=model_cfg.get("use_4bit", False),
            use_8bit=model_cfg.get("use_8bit", False),
        )
        logger.info("MedGemma loaded: %s", model_id)

    def _load_clip_rag(self) -> None:
        from ..models.clip_retriever import CLIPRetriever
        from ..models.medgemma import MedGemmaModel
        clip_model = (
            self.cfg.get("models", {})
                    .get("biomedclip", {})
                    .get("model_name", "biomedclip")
        )
        self._retriever = CLIPRetriever.load(model_name=clip_model)
        self._try_load_index("clip")
        self._vlm = MedGemmaModel.load(
            use_4bit=self.cfg.get("models", {}).get("medgemma", {}).get("use_4bit", False)
        )
        logger.info("CLIP-RAG ready (retriever=%s).", clip_model)

    def _load_colpali_rag(self) -> None:
        from ..models.colpali_retriever import ColPaliRetriever
        from ..models.medgemma import MedGemmaModel
        self._retriever = ColPaliRetriever.load()
        self._try_load_index("colpali")
        self._vlm = MedGemmaModel.load(
            use_4bit=self.cfg.get("models", {}).get("medgemma", {}).get("use_4bit", False)
        )
        logger.info("ColPali-RAG ready.")

    def _try_load_index(self, kind: str) -> None:
        base = Path(self.cfg.get("rag", {}).get("index_path", "data/index"))
        if kind == "clip":
            idx = base / "clip_index.pkl"
            if idx.exists():
                self._retriever.load_index(idx)
                logger.info("CLIP index loaded (%d entries).", self._retriever.index_size)
            else:
                logger.warning(
                    "No CLIP index at '%s'. Zero-shot fallback. "
                    "Run: python scripts/build_index.py --model clip", idx
                )
        else:
            idx = base / "colpali_index"
            if Path(str(idx) + ".npy").exists():
                self._retriever.load_index(idx)
                logger.info("ColPali index loaded (%d entries).", self._retriever.index_size)
            else:
                logger.warning(
                    "No ColPali index at '%s'. Zero-shot fallback. "
                    "Run: python scripts/build_index.py --model colpali", idx
                )

    # ── Primary public interface ───────────────────────────────────────────────

    def generate(
        self,
        image:           Union[Image.Image, str],
        prompt_template: Optional[str] = None,
        max_new_tokens:  int  = 512,
        top_k:           int  = 3,
        return_context:  bool = False,
        study_id:        str  = "",
        reference:       str  = "",
    ) -> ReportResult:
        """
        Generate a structured radiology report for a single CXR image.

        Parameters
        ----------
        image : PIL.Image.Image | str
            CXR image object or filesystem path.
        prompt_template : str, optional
            Override the default prompt. Pass any key from
            src.models.prompts.REPORT_PROMPTS, e.g.
            "medgemma_report_v2_detailed".
        max_new_tokens : int
            Maximum output tokens. 512 covers a full Findings + Impression.
        top_k : int
            Number of similar cases to retrieve (RAG modes only).
        return_context : bool
            When True the result includes retrieved context
            (useful for demo display and debugging).
        study_id : str
            Optional identifier echoed back in the result.
        reference : str
            Optional ground-truth report (for evaluation workflows).

        Returns
        -------
        ReportResult
            .findings   — FINDINGS section text
            .impression — IMPRESSION section text
            .raw        — raw model output before parsing
            .latency_s  — wall-clock time in seconds
            .full_report — "FINDINGS:\\n...\\n\\nIMPRESSION:\\n..."
        """
        if not self._is_ready:
            self.load()

        image = self._resolve_image(image)
        if image is None:
            return ReportResult(
                model=self.model_name, study_id=study_id, reference=reference,
                error="Image could not be loaded.",
            )

        t0 = time.time()
        try:
            if self.model_name in ("clip-rag", "colpali-rag") and not self.mock:
                raw_result, ctx, ids = self._generate_rag(
                    image, prompt_template, top_k, max_new_tokens
                )
            else:
                raw_result = self._vlm.generate_report(
                    image,
                    prompt_template=prompt_template,
                    max_new_tokens=max_new_tokens,
                )
                ctx, ids = "", []

            result = ReportResult(
                findings      = raw_result.get("findings",   ""),
                impression    = raw_result.get("impression", ""),
                raw           = raw_result.get("raw",        ""),
                model         = self.model_name,
                latency_s     = round(time.time() - t0, 2),
                study_id      = study_id,
                reference     = reference,
                context       = ctx if return_context else "",
                retrieved_ids = ids,
            )
        except Exception as exc:
            logger.exception(
                "Report generation failed for study '%s': %s", study_id, exc
            )
            result = ReportResult(
                model     = self.model_name,
                latency_s = round(time.time() - t0, 2),
                study_id  = study_id,
                reference = reference,
                error     = str(exc),
            )
        return result

    # backward-compat alias
    def run(self, image, **kwargs) -> dict[str, Any]:
        """Backward-compatible wrapper — returns dict instead of ReportResult."""
        return self.generate(image, **kwargs).to_dict()

    # ── Batch interface ────────────────────────────────────────────────────────

    def generate_batch(
        self,
        samples:        list[dict[str, Any]],
        max_new_tokens: int  = 512,
        top_k:          int  = 3,
        return_context: bool = False,
        show_progress:  bool = True,
        log_every:      int  = 10,
    ) -> list[ReportResult]:
        """
        Generate reports for a list of samples.

        Parameters
        ----------
        samples : list[dict]
            Each dict must have "image" (PIL or path) or "image_path".
            Optional: "study_id", "report" (ground-truth for evaluation).
        max_new_tokens, top_k, return_context
            Passed through to generate().
        show_progress : bool
            Log progress every log_every samples.

        Returns
        -------
        list[ReportResult]
        """
        if not self._is_ready:
            self.load()

        results: list[ReportResult] = []
        n = len(samples)
        errors = 0

        for i, sample in enumerate(samples):
            img = sample.get("image") or sample.get("image_path", "")
            r   = self.generate(
                image           = img,
                max_new_tokens  = max_new_tokens,
                top_k           = top_k,
                return_context  = return_context,
                study_id        = str(sample.get("study_id", f"sample_{i}")),
                reference       = str(sample.get("report", sample.get("reference", ""))),
            )
            results.append(r)
            if not r.succeeded:
                errors += 1
            if show_progress and (i + 1) % log_every == 0:
                logger.info(
                    "  [%d/%d]  errors=%d  last_latency=%.1fs",
                    i + 1, n, errors, r.latency_s,
                )

        logger.info(
            "Batch complete: %d/%d succeeded (%.0f%%).",
            n - errors, n, 100 * (n - errors) / max(n, 1),
        )
        return results

    # backward-compat alias
    def run_batch(self, samples, **kwargs) -> list[dict[str, Any]]:
        """Backward-compatible wrapper — returns list[dict]."""
        return [r.to_dict() for r in self.generate_batch(samples, **kwargs)]

    # ── Streaming ─────────────────────────────────────────────────────────────

    def generate_streaming(
        self,
        image:          Union[Image.Image, str],
        max_new_tokens: int = 512,
    ) -> Iterator[str]:
        """
        Yield the report incrementally (for Gradio streaming UIs).
        Currently yields the full report as one chunk.
        Extend with HuggingFace TextIteratorStreamer for true token streaming.
        """
        if not self._is_ready:
            self.load()
        result = self.generate(image, max_new_tokens=max_new_tokens)
        yield result.full_report

    # ── RAG generation internals ───────────────────────────────────────────────

    def _generate_rag(
        self,
        image:           Image.Image,
        prompt_template: Optional[str],
        top_k:           int,
        max_new_tokens:  int,
    ) -> tuple[dict, str, list[str]]:
        """Retrieve top-K → build context → generate report."""
        context = ""
        retrieved_ids: list[str] = []

        if self._retriever is not None and self._retriever.is_indexed:
            retrieved     = self._retriever.retrieve(query_image=image, top_k=top_k)
            context       = self._retriever.build_context_from_results(retrieved)
            retrieved_ids = [r.get("study_id", "") for r in retrieved]
            logger.debug(
                "Retrieved %d cases  top_score=%.3f",
                len(retrieved),
                retrieved[0].get("score", 0.0) if retrieved else 0.0,
            )
        else:
            mode = self.model_name.split("-")[0].upper()
            logger.warning("%s-RAG: retriever not indexed — zero-shot fallback.", mode)

        raw_result = self._vlm.generate_report(
            image,
            prompt_template=prompt_template,
            context=context or None,
            max_new_tokens=max_new_tokens,
        )
        return raw_result, context, retrieved_ids

    # ── Static utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_image(image: Union[Image.Image, str, None]) -> Optional[Image.Image]:
        """Return RGB PIL Image from any input type; None if unresolvable."""
        if image is None:
            return None
        if isinstance(image, str):
            p = Path(image)
            if not p.exists():
                logger.warning("Image path does not exist: %s", image)
                return None
            try:
                return Image.open(p).convert("RGB")
            except Exception as exc:
                logger.warning("Failed to open image '%s': %s", image, exc)
                return None
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        logger.warning("Unsupported image type: %s", type(image))
        return None

    @staticmethod
    def compare(
        result_a:  Union[ReportResult, dict],
        result_b:  Union[ReportResult, dict],
        reference: Optional[str] = None,
    ) -> None:
        """
        Pretty-print a side-by-side comparison of two model outputs
        with optional RadGraph heuristic F1 and ROUGE-L scores.
        """
        def _get(r, k):
            return r.get(k, "") if isinstance(r, dict) else getattr(r, k, "")

        W = 66
        SEP = "─" * W

        def _block(result) -> None:
            model = _get(result, "model") or "?"
            lat   = _get(result, "latency_s")
            err   = _get(result, "error")
            print(f"\n{SEP}")
            print(f"  {model}  (latency: {lat}s)")
            print(SEP)
            if err:
                print(f"  [!]  ERROR: {err}")
            else:
                f   = _get(result, "findings")   or "(empty)"
                imp = _get(result, "impression") or "(empty)"
                print(f"FINDINGS:\n{f}\n")
                print(f"IMPRESSION:\n{imp}")

        print(f"\n{'=' * W}")
        print("  REPORT GENERATION COMPARISON")
        print("=" * W)
        _block(result_a)
        _block(result_b)

        if reference:
            print(f"\n{SEP}")
            print("  REFERENCE (ground truth)")
            print(SEP)
            print(reference)

        # Quick metric peek
        try:
            from ..evaluation.metrics import _radgraph_heuristic_fallback, compute_rouge
            print(f"\n{SEP}")
            print(f"  {'Model':<16}  RadGraph F1 (heuristic)   ROUGE-L")
            print(SEP)
            for label, res in [("Model A", result_a), ("Model B", result_b)]:
                raw = _get(res, "raw") or (
                    _get(res, "findings") + " " + _get(res, "impression")
                )
                if raw.strip() and reference:
                    rg = _radgraph_heuristic_fallback([raw], [reference])
                    rl = compute_rouge([raw], [reference])
                    print(f"  {label:<16}  {rg['radgraph_f1']:>22.4f}   {rl['rouge_l']:.4f}")
        except Exception:
            pass
        print(f"{SEP}\n")

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __call__(self, image, **kwargs) -> ReportResult:
        """Shorthand: report = pipeline(image)."""
        return self.generate(image, **kwargs)

    def __repr__(self) -> str:
        return (
            f"ReportGenerationPipeline("
            f"model={self.model_name!r}, "
            f"ready={self._is_ready}, "
            f"mock={self.mock})"
        )
