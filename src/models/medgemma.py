"""
src/models/medgemma.py
======================
MedGemma-4B-it wrapper for:
  1. Zero-shot & fine-tuned chest X-ray report generation
  2. RAG-augmented clinical QA

Model: google/medgemma-4b-it  (gated — requires HF_TOKEN + accepted ToS)
Architecture: Gemma3-4B language model + MedSigLIP vision encoder

GPU requirements:
  bfloat16 full:  ~9 GB VRAM
  4-bit (QLoRA):  ~5 GB VRAM
  CPU (slow):     ~16 GB RAM
"""

from __future__ import annotations

import gc
import logging
import os
import re
from typing import Any, Optional, Union

import torch
from PIL import Image

from .base import BaseVLM
from .prompts import get_report_prompt, get_qa_prompt
from ..data.report_parser import ReportParser

logger = logging.getLogger(__name__)


class MedGemmaModel(BaseVLM):
    """
    Wrapper around MedGemma-4B-it (or MedGemma-1.5-4B-it) for:
      - Structured chest X-ray report generation
      - Clinical visual question answering (with optional RAG context)

    Usage (GPU):
        model = MedGemmaModel.load()
        report = model.generate_report(pil_image)

    Usage (4-bit, low VRAM):
        model = MedGemmaModel.load(use_4bit=True)

    Usage (mock, no GPU needed):
        model = MedGemmaModel.load(mock=True)
    """

    DEFAULT_MODEL_ID = "google/medgemma-4b-it"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,   # None = auto (float16 on Windows, bfloat16 elsewhere)
        use_4bit: bool = False,
        use_8bit: bool = False,
        report_prompt_name: str = "medgemma_report_v1",
        qa_prompt_name: str = "rag",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype or self._pick_dtype()
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.report_prompt_name = report_prompt_name
        self.qa_prompt_name = qa_prompt_name

        self._model = None
        self._processor = None
        self._parser = ReportParser()

    @staticmethod
    def _pick_dtype() -> torch.dtype:
        """
        Choose the safest dtype for the current platform.
        - Windows + CUDA: float16  (bfloat16 has known issues on some Windows drivers)
        - Linux  + Ampere+ GPU: bfloat16  (preferred — wider dynamic range)
        - Anything else: float16
        """
        import sys as _sys
        if torch.cuda.is_available():
            if _sys.platform == "win32":
                return torch.float16
            # Check if current GPU supports bfloat16 natively (Ampere = sm_80+)
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 8:
                return torch.bfloat16
            return torch.float16
        return torch.float32   # CPU

    # ── Loading ───────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        use_4bit: bool = False,
        use_8bit: bool = False,
        hf_token: Optional[str] = None,
        mock: bool = False,
    ) -> "MedGemmaModel":
        """
        Factory method: load MedGemma model weights.

        Args:
            model_id:  HuggingFace model ID.
            device:    "cuda" | "mps" | "cpu". Auto-detected if None.
            use_4bit:  Enable bitsandbytes 4-bit quantisation (~5 GB VRAM).
            use_8bit:  Enable bitsandbytes 8-bit quantisation (~7 GB VRAM).
            hf_token:  HuggingFace access token. Falls back to HF_TOKEN env var.
            mock:      Return a mock model (no weights loaded — for testing).
        """
        from ..models.base import MockVLM  # avoid circular at module level

        if mock or os.environ.get("MOCK_MODELS", "").lower() in ("true", "1"):
            logger.info("MedGemma: loading in MOCK mode (no GPU required).")
            m = cls(model_id=model_id)
            m._mock = True
            return m

        if device is None:
            device = cls._detect_device()

        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            logger.warning(
                "HF_TOKEN not set. MedGemma is a gated model — you MUST accept "
                "the Health AI terms at huggingface.co/google/medgemma-4b-it "
                "and provide your HF token via HF_TOKEN env var or hf_token arg."
            )

        instance = cls(
            model_id=model_id,
            device=device,
            use_4bit=use_4bit,
            use_8bit=use_8bit,
        )
        instance._mock = False
        instance._load_weights(token=token)
        return instance

    def _load_weights(self, token=None):
        """
        Load MedGemma with GPU-memory-aware strategy ladder.

        Strategy 1: 4-bit NF4 + GPU+CPU offload with max_memory budget
                    Works on 4 GB VRAM (RTX 3050) — offloads overflow to RAM.
        Strategy 2: 4-bit NF4 on CPU only — slow but always works.
        Strategy 3: float16 full-precision on CPU — last resort.

        For RTX 3050 (4 GB VRAM):
          ~2.5 GB quantized weights on GPU, ~4 GB overflow layers on CPU RAM.
          Inference ~30-90 s/image due to GPU<->CPU transfers.
        """
        from transformers import AutoProcessor, AutoModelForImageTextToText
        import transformers as _tv

        logger.info("Loading MedGemma processor: %s", self.model_id)
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, token=token,
        )

        # transformers >= 4.51 uses dtype= instead of torch_dtype=
        _tv_ver = tuple(int(x) for x in _tv.__version__.split(".")[:2])
        dtype_key = "dtype" if _tv_ver >= (4, 51) else "torch_dtype"

        compute_dtype = torch.float16  # float16 on Windows CUDA

        # ── Detect usable VRAM and RAM ────────────────────────────────────────
        gpu_ok = torch.cuda.is_available()
        if gpu_ok:
            total_vram   = torch.cuda.get_device_properties(0).total_memory
            usable_vram  = int(total_vram * 0.85) - 256 * 1024 * 1024
            vram_str     = f"{max(usable_vram,0)/(1024**3):.1f}GiB"
            logger.info(
                "GPU: %s | total VRAM %.1f GiB | model budget %s",
                torch.cuda.get_device_name(0),
                total_vram / (1024**3), vram_str,
            )
            try:
                import psutil
                ram_avail = psutil.virtual_memory().available
                ram_str   = f"{max(int(ram_avail*0.8)-1024**3, 2*1024**3)/(1024**3):.0f}GiB"
            except ImportError:
                ram_str = "10GiB"
            max_memory = {0: vram_str, "cpu": ram_str}
        else:
            max_memory = {"cpu": "12GiB"}

        logger.info("Memory budget: %s", max_memory)

        # ── BitsAndBytes 4-bit config ─────────────────────────────────────────
        bnb_cfg = None
        try:
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        except Exception as e:
            logger.warning("BitsAndBytesConfig unavailable: %s", e)

        # ── Strategy 1: 4-bit + GPU/CPU offload ──────────────────────────────
        if bnb_cfg is not None and gpu_ok:
            try:
                logger.info("Strategy 1: 4-bit NF4 + GPU/CPU offload ...")
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    **{dtype_key: compute_dtype},
                    quantization_config=bnb_cfg,
                    device_map="auto",
                    max_memory=max_memory,
                    token=token,
                    low_cpu_mem_usage=True,
                )
                self._model.eval()
                logger.info(
                    "Strategy 1 OK. Device map: %s",
                    getattr(self._model, "hf_device_map", "N/A"),
                )
                return
            except Exception as e1:
                logger.warning("Strategy 1 failed: %s", e1)

        # ── Strategy 2: 4-bit on CPU only ────────────────────────────────────
        if bnb_cfg is not None:
            try:
                logger.info("Strategy 2: 4-bit NF4 on CPU ...")
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    **{dtype_key: compute_dtype},
                    quantization_config=bnb_cfg,
                    device_map={"": "cpu"},
                    token=token,
                    low_cpu_mem_usage=True,
                )
                self._model.eval()
                logger.info("Strategy 2 OK. MedGemma on CPU (4-bit).")
                return
            except Exception as e2:
                logger.warning("Strategy 2 failed: %s", e2)

        # ── Strategy 3: float16 full-precision CPU ────────────────────────────
        logger.info("Strategy 3: float16 full-precision on CPU (slow) ...")
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            **{dtype_key: torch.float16},
            device_map={"": "cpu"},
            token=token,
            low_cpu_mem_usage=True,
        )
        self._model.eval()
        logger.info("Strategy 3 OK. MedGemma on CPU (float16).")

    def generate_report(
        self,
        image: Image.Image,
        prompt_template: Optional[str] = None,
        context: Optional[str] = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        repetition_penalty: float = 1.1,
    ) -> dict[str, str]:
        """
        Generate a structured chest X-ray report.

        Args:
            image:           PIL Image (RGB).
            prompt_template: Override prompt template name.
            context:         Optional retrieved reports for RAG-augmented generation.
            max_new_tokens:  Maximum output length in tokens.
            do_sample:       Greedy (False) vs. sampling (True).
            temperature:     Sampling temperature (only used when do_sample=True).
            repetition_penalty: Penalise repeated tokens.

        Returns:
            dict with keys: "findings", "impression", "raw"
        """
        if getattr(self, "_mock", False):
            from .base import MockVLM
            return MockVLM().generate_report(image)

        # Select prompt
        pname = prompt_template or self.report_prompt_name
        if context:
            prompt = get_report_prompt("medgemma_report_rag", context=context)
        else:
            prompt = get_report_prompt(pname)

        # Build message
        raw_output = self._run_inference(
            image=image,
            text_prompt=prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        # Parse into sections
        sections = self._parser.parse(raw_output)
        return {
            "findings":   sections.findings,
            "impression": sections.impression,
            "raw":        raw_output,
        }

    # ── QA ────────────────────────────────────────────────────────────────────

    def answer_question(
        self,
        image: Image.Image,
        question: str,
        context: Optional[str] = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> str:
        """
        Answer a clinical question about a chest X-ray.

        Args:
            image:    PIL Image of the CXR.
            question: Clinical question string.
            context:  Retrieved reports / knowledge base context for RAG.

        Returns:
            Answer string.
        """
        if getattr(self, "_mock", False):
            from .base import MockVLM
            return MockVLM().answer_question(image, question, context)

        if context:
            prompt = get_qa_prompt("rag", question=question, context=context)
        else:
            prompt = get_qa_prompt("zero_shot", question=question)

        return self._run_inference(
            image=image,
            text_prompt=prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

    # ── Core inference ────────────────────────────────────────────────────────

    @staticmethod
    def _blank_cxr(size: int = 448) -> "Image.Image":
        """
        Create a blank grey placeholder image for samples with no CXR.
        MedGemma still runs but will produce a generic 'normal' report.
        """
        import numpy as np
        arr = np.full((size, size, 3), 128, dtype=np.uint8)
        return Image.fromarray(arr)

    def _get_model_device(self) -> str:
        """
        Return the device string for the first parameter of the model.
        Works for both standard models and CPU-offloaded multi-device models.
        """
        try:
            return str(next(self._model.parameters()).device)
        except Exception:
            return "cpu"

    def _run_inference(
        self,
        image: Optional["Image.Image"],
        text_prompt: str,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> str:
        """
        Single forward pass through MedGemma.

        Handles:
        - CPU-offloaded models (device_map with mixed GPU/CPU layers)
        - None image (uses blank placeholder)
        - Windows float16 dtype
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Call MedGemmaModel.load() first.")

        # Use blank placeholder if no image available (text metrics still work)
        if image is None:
            logger.debug("No image provided — using blank placeholder.")
            image = self._blank_cxr()

        image = image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text":  text_prompt},
                ],
            }
        ]

        prompt = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )

        # Move tensors individually
        try:
            if hasattr(inputs, "to"):
                primary_device = self._get_model_device()
                inputs = inputs.to(primary_device)
        except Exception as e:
            logger.warning("Could not move inputs to device: %s", e)

        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        generated_ids = outputs[0][input_len:]

        decoded = self._processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        return decoded.strip()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self.model_id.split("/")[-1]

    @property
    def is_loaded(self) -> bool:
        return getattr(self, "_mock", False) or self._model is not None

    # ── Fine-tuning entrypoint ────────────────────────────────────────────────

    def prepare_for_finetuning(
        self,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[list[str]] = None,
    ):
        """
        Attach LoRA adapters to the model for QLoRA fine-tuning.
        Call this AFTER load() and BEFORE creating a Trainer.

        Args:
            lora_rank:       LoRA rank r. Higher = more capacity, more memory.
            lora_alpha:      LoRA alpha scaling factor.
            lora_dropout:    Dropout in LoRA layers.
            target_modules:  Which linear layers to adapt. Defaults to Q/K/V/O.

        Returns:
            PEFT model ready for training.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        from peft import LoraConfig, get_peft_model, TaskType

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        self._model = get_peft_model(self._model, lora_config)
        self._model.print_trainable_parameters()
        logger.info("LoRA adapters attached. Ready for fine-tuning.")
        return self._model

    # ── Utilities ─────────────────────────────────────────────────────────────

    def unload(self) -> None:
        """Free GPU memory."""
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("MedGemma unloaded from memory.")

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        logger.warning("No GPU detected — running MedGemma on CPU (very slow).")
        return "cpu"

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return (
            f"MedGemmaModel(id={self.model_id!r}, "
            f"device={self.device!r}, "
            f"4bit={self.use_4bit}, status={status})"
        )
