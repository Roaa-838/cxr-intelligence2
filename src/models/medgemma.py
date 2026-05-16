"""
src/models/medgemma.py
======================
MedGemma-4B-it wrapper for chest X-ray report generation and QA.
GPU strategy: 4-bit NF4 + GPU+CPU offload for 4 GB VRAM (RTX 3050).
"""

from __future__ import annotations
import gc, logging, os, re
from typing import Any, Optional, Union
import torch
from PIL import Image

from .base import BaseVLM
from .prompts import get_report_prompt, get_qa_prompt
from ..data.report_parser import ReportParser

logger = logging.getLogger(__name__)


class MedGemmaModel(BaseVLM):

    DEFAULT_MODEL_ID = "google/medgemma-4b-it"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
        use_4bit: bool = False,
        use_8bit: bool = False,
        report_prompt_name: str = "medgemma_report_v1",
        qa_prompt_name: str = "rag",
    ) -> None:
        self.model_id           = model_id
        self.device             = device
        self.torch_dtype        = torch_dtype or self._pick_dtype()
        self.use_4bit           = use_4bit
        self.use_8bit           = use_8bit
        self.report_prompt_name = report_prompt_name
        self.qa_prompt_name     = qa_prompt_name
        self._model             = None
        self._processor         = None
        self._parser            = ReportParser()

    # ── dtype selection ───────────────────────────────────────────────────────

    @staticmethod
    def _pick_dtype() -> torch.dtype:
        import sys as _sys
        if torch.cuda.is_available():
            return torch.float16          # float16 on Windows CUDA
        return torch.float32              # CPU

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        logger.warning("No GPU detected — running MedGemma on CPU (very slow).")
        return "cpu"

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
        if mock or os.environ.get("MOCK_MODELS", "").lower() in ("true", "1"):
            logger.info("MedGemma: MOCK mode.")
            m = cls(model_id=model_id)
            m._mock = True
            return m

        if device is None:
            device = cls._detect_device()

        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            logger.warning(
                "HF_TOKEN not set. MedGemma is gated — accept terms at "
                "huggingface.co/google/medgemma-4b-it and set HF_TOKEN."
            )

        instance = cls(model_id=model_id, device=device, use_4bit=use_4bit, use_8bit=use_8bit)
        instance._mock = False
        instance._load_weights(token=token)
        return instance

    def _load_weights(self, token=None):
        """
        3-strategy loader for RTX 3050 (4 GB VRAM):
          1. 4-bit NF4 + GPU+CPU offload  (fits in 4 GB + RAM)
          2. 4-bit NF4 CPU-only
          3. float16 CPU full-precision
        """
        from transformers import AutoProcessor, AutoModelForImageTextToText

        logger.info("Loading MedGemma processor: %s", self.model_id)
        self._processor = AutoProcessor.from_pretrained(self.model_id, token=token)

        compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        gpu_ok        = torch.cuda.is_available()

        # Memory budgets
        if gpu_ok:
            total_vram   = torch.cuda.get_device_properties(0).total_memory
            usable_vram  = int(total_vram * 0.85) - 256 * 1024 * 1024
            vram_str     = f"{max(usable_vram, 0) / (1024**3):.1f}GiB"
            logger.info("GPU: %s | VRAM budget %s", torch.cuda.get_device_name(0), vram_str)
            try:
                import psutil
                ram_avail = psutil.virtual_memory().available
                ram_str   = f"{max(int(ram_avail * 0.8) - 1024**3, 2 * 1024**3) / (1024**3):.0f}GiB"
            except ImportError:
                ram_str = "10GiB"
            max_memory = {0: vram_str, "cpu": ram_str}
        else:
            max_memory = {"cpu": "12GiB"}

        logger.info("Memory budget: %s", max_memory)

        # BitsAndBytes config
        bnb_cfg = None
        try:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        except Exception as e:
            logger.warning("BitsAndBytesConfig unavailable: %s", e)

        # Strategy 1: 4-bit + GPU/CPU offload
        if bnb_cfg is not None and gpu_ok:
            try:
                logger.info("Strategy 1: 4-bit NF4 + GPU/CPU offload ...")
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=compute_dtype,
                    quantization_config=bnb_cfg,
                    device_map={"": 0},
                    token=token,
                    low_cpu_mem_usage=True,
                )
                self._model.eval()
                logger.info("Strategy 1 OK. Device map: %s",
                            getattr(self._model, "hf_device_map", "N/A"))
                return
            except Exception as e1:
                logger.warning("Strategy 1 failed: %s", e1)

        # Strategy 2: 4-bit CPU-only
        if bnb_cfg is not None:
            try:
                logger.info("Strategy 2: 4-bit NF4 on CPU ...")
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=compute_dtype,
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

        # Strategy 3: float16 CPU
        logger.info("Strategy 3: float16 full-precision on CPU ...")
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map={"": "cpu"},
            token=token,
            low_cpu_mem_usage=True,
        )
        self._model.eval()
        logger.info("Strategy 3 OK. MedGemma on CPU (float16).")

    # ── Report generation ─────────────────────────────────────────────────────

    def generate_report(
        self,
        image: Optional[Image.Image],
        prompt_template: Optional[str] = None,
        context: Optional[str] = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        repetition_penalty: float = 1.1,
        **kwargs,
    ) -> dict[str, str]:
        if getattr(self, "_mock", False):
            from .base import MockVLM
            return MockVLM().generate_report(image)

        pname  = prompt_template or self.report_prompt_name
        prompt = (get_report_prompt("medgemma_report_rag", context=context)
                  if context else get_report_prompt(pname))

        raw = self._run_inference(
            image=image, text_prompt=prompt,
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature, repetition_penalty=repetition_penalty,
        )
        sections = self._parser.parse(raw)
        return {"findings": sections.findings, "impression": sections.impression, "raw": raw}

    # ── QA ────────────────────────────────────────────────────────────────────

    def answer_question(
        self,
        image: Optional[Image.Image],
        question: str,
        context: Optional[str] = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        **kwargs,
    ) -> str:
        if getattr(self, "_mock", False):
            from .base import MockVLM
            return MockVLM().answer_question(image, question, context)

        prompt = (get_qa_prompt("rag", question=question, context=context)
                  if context else get_qa_prompt("zero_shot", question=question))
        return self._run_inference(
            image=image, text_prompt=prompt,
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature,
        )

    # ── Core inference ────────────────────────────────────────────────────────

    @staticmethod
    def _blank_cxr(size: int = 448) -> Image.Image:
        import numpy as np
        return Image.fromarray(np.full((size, size, 3), 128, dtype=np.uint8))

    def _get_model_device(self) -> str:
        try:
            return str(next(self._model.parameters()).device)
        except Exception:
            return "cpu"

    def _run_inference(
        self,
        image: Optional[Image.Image],
        text_prompt: str,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> str:
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Call MedGemmaModel.load() first.")

        if image is None:
            image = self._blank_cxr()

        image = image.convert("RGB")

        # Build prompt string via chat template (MedGemma requires this format)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  text_prompt},
        ]}]
        prompt_str = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )

        # Tokenise image + text together
        inputs = self._processor(
            text=prompt_str, images=image, return_tensors="pt",
        )

        # Move to primary device (handles GPU+CPU offload models)
        try:
            primary = self._get_model_device()
            inputs  = inputs.to(primary)
        except Exception:
            pass

        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens":    max_new_tokens,
            "do_sample":         do_sample,
            "repetition_penalty": repetition_penalty,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        decoded = self._processor.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True,
        )
        return decoded.strip()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self.model_id.split("/")[-1]

    @property
    def is_loaded(self) -> bool:
        return getattr(self, "_mock", False) or self._model is not None

    def prepare_for_finetuning(self, lora_rank=16, lora_alpha=32,
                               lora_dropout=0.05, target_modules=None):
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        from peft import LoraConfig, get_peft_model, TaskType
        if target_modules is None:
            target_modules = ["q_proj","k_proj","v_proj","o_proj",
                              "gate_proj","up_proj","down_proj"]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=lora_rank,
            lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=target_modules, bias="none",
        )
        self._model = get_peft_model(self._model, lora_config)
        self._model.print_trainable_parameters()
        return self._model

    def unload(self) -> None:
        del self._model, self._processor
        self._model = self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        return (f"MedGemmaModel(id={self.model_id!r}, "
                f"device={self.device!r}, loaded={self.is_loaded})")
