"""src/models/base.py — Abstract base classes for all models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union
from PIL import Image
import torch


class BaseVLM(ABC):
    """
    Abstract base for vision-language models used in report generation and QA.
    
    Subclasses: MedGemmaModel, MockVLM
    """

    @abstractmethod
    def generate_report(
        self,
        image: Image.Image,
        prompt_template: Optional[str] = None,
    ) -> dict[str, str]:
        """
        Generate a structured radiology report from a CXR image.
        
        Returns:
            dict with keys: "findings", "impression", "raw"
        """
        ...

    @abstractmethod
    def answer_question(
        self,
        image: Image.Image,
        question: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Answer a clinical question about a CXR image, optionally with
        retrieved context (for RAG mode).
        
        Returns:
            Plain-text answer string.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether model weights are loaded."""
        ...


class BaseRetriever(ABC):
    """
    Abstract base for embedding + retrieval models.
    
    Subclasses: CLIPRetriever, ColPaliRetriever
    """

    @abstractmethod
    def embed_image(self, image: Image.Image) -> torch.Tensor:
        """Return a normalised embedding vector for an image."""
        ...

    @abstractmethod
    def embed_text(self, text: str) -> torch.Tensor:
        """Return a normalised embedding vector for a text string."""
        ...

    @abstractmethod
    def retrieve(
        self,
        query_image: Optional[Image.Image],
        query_text: Optional[str],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top_k most relevant documents from the index.
        
        Returns:
            list of dicts with keys: study_id, score, report, image_path
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    @abstractmethod
    def is_indexed(self) -> bool:
        """Whether a knowledge base has been indexed."""
        ...


class MockVLM(BaseVLM):
    """
    Mock VLM for testing without GPU / model weights.
    Returns plausible-looking reports deterministically.
    """

    MOCK_REPORTS = [
        {
            "findings": (
                "The heart size is normal. The mediastinum is not widened. "
                "The lungs are clear bilaterally with no focal consolidation. "
                "No pleural effusion or pneumothorax is identified. "
                "The bony structures are intact."
            ),
            "impression": "No acute cardiopulmonary process.",
        },
        {
            "findings": (
                "Heart size is mildly enlarged. There is mild pulmonary vascular "
                "congestion. Small bilateral pleural effusions are present. "
                "No pneumothorax identified."
            ),
            "impression": (
                "Cardiomegaly with mild congestive heart failure. "
                "Small bilateral pleural effusions."
            ),
        },
        {
            "findings": (
                "There is increased opacity in the right lower lobe consistent "
                "with consolidation. The heart size is normal. "
                "No pleural effusion on the left is identified."
            ),
            "impression": (
                "Right lower lobe consolidation. Pneumonia is a primary consideration. "
                "Clinical correlation recommended."
            ),
        },
    ]

    def __init__(self, report_idx: int = 0) -> None:
        self._report_idx = report_idx % len(self.MOCK_REPORTS)

    def generate_report(self, image, prompt_template=None, **kwargs):
        r = self.MOCK_REPORTS[self._report_idx]
        return {
            "findings":   r["findings"],
            "impression": r["impression"],
            "raw":        f"FINDINGS:\n{r['findings']}\n\nIMPRESSION:\n{r['impression']}",
        }

    def answer_question(self, image, question, context=None, **kwargs):
        return (
            f"[Mock answer to: '{question}']\n"
            "Based on the chest X-ray, no acute abnormality is identified. "
            "The findings are within normal limits."
        )

    @property
    def model_name(self) -> str:
        return "MockVLM"

    @property
    def is_loaded(self) -> bool:
        return True


class MockRetriever(BaseRetriever):
    """Mock retriever for testing."""

    def __init__(self, embedding_dim: int = 512) -> None:
        self.embedding_dim = embedding_dim
        self._indexed = False

    def embed_image(self, image):
        return torch.randn(self.embedding_dim)

    def embed_text(self, text):
        return torch.randn(self.embedding_dim)

    def build_index(self, studies, batch_size=4):
        self._indexed = True

    def retrieve(self, query_image=None, query_text=None, top_k=5):
        return [
            {
                "study_id": f"synth_{i:05d}",
                "score": float(1.0 - i * 0.1),
                "report": (
                    "FINDINGS:\nThe lungs are clear. Heart size is normal. "
                    "No pleural effusion or pneumothorax.\n\n"
                    "IMPRESSION:\nNo acute cardiopulmonary process."
                ),
                "image_path": "",
                "rank": i + 1,
            }
            for i in range(min(top_k, 3))
        ]

    @property
    def model_name(self) -> str:
        return "MockRetriever"

    @property
    def is_indexed(self) -> bool:
        return self._indexed
