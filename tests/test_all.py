"""
tests/test_all.py
==================
Comprehensive pytest test suite for the CXR Intelligence System.
All tests run in MOCK mode — no GPU or model weights required.

Run:
    pytest tests/ -v
    pytest tests/ -v --tb=short -q   # quiet
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["MOCK_MODELS"] = "true"


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_cxr() -> Image.Image:
    """Synthetic 448×448 chest X-ray-like image."""
    arr = np.random.randint(30, 200, (448, 448), dtype=np.uint8)
    return Image.fromarray(arr).convert("RGB")


@pytest.fixture
def sample_report() -> str:
    return (
        "FINDINGS:\nThe heart size is normal. The mediastinum is not widened. "
        "The lungs are clear bilaterally. No pleural effusion or pneumothorax. "
        "No acute osseous abnormality.\n\n"
        "IMPRESSION:\nNo acute cardiopulmonary process."
    )


@pytest.fixture
def sample_reports() -> list[str]:
    return [
        "FINDINGS:\nMild cardiomegaly. No pleural effusion. Clear lungs.\n\nIMPRESSION:\nCardiomegaly.",
        "FINDINGS:\nRight lower lobe consolidation. Heart normal.\n\nIMPRESSION:\nPneumonia.",
        "FINDINGS:\nLungs clear. No effusion. Heart normal.\n\nIMPRESSION:\nNo acute process.",
    ]


@pytest.fixture
def synthetic_dataset():
    from src.data.dataset import SyntheticCXRDataset
    return SyntheticCXRDataset(n_samples=10)


# ─────────────────────────────────────────────────────────────────────────────
#  Data — ReportParser
# ─────────────────────────────────────────────────────────────────────────────

class TestReportParser:
    def test_parse_standard_sections(self, sample_report):
        from src.data.report_parser import ReportParser
        parser = ReportParser()
        s = parser.parse(sample_report)
        assert s.has_findings
        assert s.has_impression
        assert "heart" in s.findings.lower()
        assert "cardiopulmonary" in s.impression.lower()

    def test_parse_impression_only(self):
        from src.data.report_parser import ReportParser
        parser = ReportParser()
        s = parser.parse("IMPRESSION:\nNo acute process.")
        assert s.has_impression
        assert "acute" in s.impression.lower()

    def test_parse_unsectioned(self):
        from src.data.report_parser import ReportParser
        parser = ReportParser()
        s = parser.parse("Lungs are clear. Heart is normal. No pneumothorax.")
        assert s.raw != ""

    def test_parse_empty(self):
        from src.data.report_parser import ReportParser
        parser = ReportParser()
        s = parser.parse("")
        assert s.findings == ""
        assert s.impression == ""

    def test_clinical_entity_extraction(self, sample_report):
        from src.data.report_parser import extract_findings_presence
        entities = extract_findings_presence(sample_report)
        assert isinstance(entities, dict)
        assert "pneumothorax" in entities
        assert "no_finding" in entities


# ─────────────────────────────────────────────────────────────────────────────
#  Data — Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_medgemma_preprocessor(self, dummy_cxr):
        from src.data.preprocessing import CXRPreprocessor
        prep = CXRPreprocessor.for_medgemma()
        tensor = prep(dummy_cxr)
        assert tensor.shape == (3, 448, 448)
        assert tensor.dtype.is_floating_point

    def test_clip_preprocessor(self, dummy_cxr):
        from src.data.preprocessing import CXRPreprocessor
        prep = CXRPreprocessor.for_clip()
        tensor = prep(dummy_cxr)
        assert tensor.shape == (3, 224, 224)

    def test_colpali_preprocessor(self, dummy_cxr):
        from src.data.preprocessing import CXRPreprocessor
        prep = CXRPreprocessor.for_colpali()
        tensor = prep(dummy_cxr)
        assert tensor.shape == (3, 448, 448)

    def test_process_pil_returns_image(self, dummy_cxr):
        from src.data.preprocessing import CXRPreprocessor
        prep = CXRPreprocessor.for_medgemma()
        pil = prep.process_pil(dummy_cxr)
        assert isinstance(pil, Image.Image)
        assert pil.size == (448, 448)

    def test_validate_cxr(self, dummy_cxr):
        from src.data.preprocessing import validate_cxr_image
        result = validate_cxr_image(dummy_cxr)
        assert "valid" in result
        assert "warnings" in result
        assert isinstance(result["warnings"], list)

    def test_rgba_conversion(self):
        from src.data.preprocessing import CXRPreprocessor
        rgba = Image.new("RGBA", (224, 224), (128, 128, 128, 200))
        prep = CXRPreprocessor.for_clip()
        tensor = prep(rgba)
        assert tensor.shape == (3, 224, 224)


# ─────────────────────────────────────────────────────────────────────────────
#  Data — Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticDataset:
    def test_length(self, synthetic_dataset):
        assert len(synthetic_dataset) == 10

    def test_getitem(self, synthetic_dataset):
        s = synthetic_dataset[0]
        assert s.study_id.startswith("synth_")
        assert len(s.report) > 20
        assert s.image is not None
        assert isinstance(s.image, Image.Image)

    def test_findings_impression_split(self, synthetic_dataset):
        s = synthetic_dataset[0]
        assert len(s.findings) > 0
        assert len(s.impression) > 0


# ─────────────────────────────────────────────────────────────────────────────
#  Models — Mock models
# ─────────────────────────────────────────────────────────────────────────────

class TestMockModels:
    def test_mock_vlm_report(self, dummy_cxr):
        from src.models.base import MockVLM
        vlm = MockVLM()
        result = vlm.generate_report(dummy_cxr)
        assert "findings" in result
        assert "impression" in result
        assert len(result["findings"]) > 10

    def test_mock_vlm_qa(self, dummy_cxr):
        from src.models.base import MockVLM
        vlm = MockVLM()
        answer = vlm.answer_question(dummy_cxr, "Is there a pleural effusion?")
        assert isinstance(answer, str)
        assert len(answer) > 5

    def test_mock_retriever(self, dummy_cxr):
        from src.models.base import MockRetriever
        ret = MockRetriever()
        results = ret.retrieve(query_image=dummy_cxr, query_text="pleural effusion", top_k=3)
        assert len(results) <= 3
        assert all("score" in r for r in results)

    def test_medgemma_mock_mode(self, dummy_cxr):
        from src.models.medgemma import MedGemmaModel
        model = MedGemmaModel.load(mock=True)
        assert model.is_loaded
        result = model.generate_report(dummy_cxr)
        assert "findings" in result

    def test_clip_mock_mode(self, dummy_cxr):
        from src.models.clip_retriever import CLIPRetriever
        import torch
        ret = CLIPRetriever.load(mock=True)
        emb = ret.embed_image(dummy_cxr)
        assert emb.shape[-1] == 512  # BiomedCLIP embedding dim

    def test_colpali_mock_mode(self, dummy_cxr):
        from src.models.colpali_retriever import ColPaliRetriever
        ret = ColPaliRetriever.load(mock=True)
        emb = ret.embed_image(dummy_cxr)
        # (N_patches, 128)
        assert emb.shape[-1] == ColPaliRetriever.EMBEDDING_DIM


# ─────────────────────────────────────────────────────────────────────────────
#  CLIP Retriever — indexing and retrieval
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIPRetriever:
    def test_build_and_retrieve(self, dummy_cxr):
        from src.models.clip_retriever import CLIPRetriever
        ret = CLIPRetriever.load(mock=True)
        studies = [
            {"image": dummy_cxr, "report": "Normal CXR.", "study_id": f"s{i}"}
            for i in range(5)
        ]
        ret.build_index(studies)
        assert ret.is_indexed
        assert ret.index_size == 5

        results = ret.retrieve(query_image=dummy_cxr, top_k=3)
        assert len(results) == 3
        assert all("study_id" in r for r in results)

    def test_save_load_index(self, dummy_cxr, tmp_path):
        from src.models.clip_retriever import CLIPRetriever
        ret = CLIPRetriever.load(mock=True)
        studies = [{"image": dummy_cxr, "report": "test", "study_id": "s0"}]
        ret.build_index(studies)

        idx_path = tmp_path / "test_clip.pkl"
        ret.save_index(idx_path)
        assert idx_path.exists()

        ret2 = CLIPRetriever.load(mock=True)
        ret2.load_index(idx_path)
        assert ret2.is_indexed

    def test_context_building(self, dummy_cxr):
        from src.models.clip_retriever import CLIPRetriever
        ret = CLIPRetriever.load(mock=True)
        results = [
            {"rank": 1, "score": 0.9, "study_id": "s1",
             "report": "Normal CXR. No effusion."},
        ]
        ctx = ret.build_context_from_results(results)
        assert "Normal" in ctx
        assert "0.900" in ctx


# ─────────────────────────────────────────────────────────────────────────────
#  ColPali Retriever
# ─────────────────────────────────────────────────────────────────────────────

class TestColPaliRetriever:
    def test_maxsim_score(self):
        import torch
        from src.models.colpali_retriever import ColPaliRetriever
        q = torch.randn(10, 128)
        d = torch.randn(1024, 128)
        score = ColPaliRetriever.maxsim_score(q, d)
        assert isinstance(score, float)

    def test_build_and_retrieve_mock(self, dummy_cxr):
        from src.models.colpali_retriever import ColPaliRetriever
        ret = ColPaliRetriever.load(mock=True)
        studies = [
            {"image": dummy_cxr, "report": "Normal CXR.", "study_id": f"s{i}"}
            for i in range(5)
        ]
        ret.build_index(studies)
        assert ret.is_indexed
        results = ret.retrieve(query_text="pleural effusion", top_k=3)
        assert len(results) <= 3


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_bleu_identical(self):
        from src.evaluation.metrics import compute_bleu
        hyps = ["The cat sat on the mat."]
        refs = ["The cat sat on the mat."]
        r = compute_bleu(hyps, refs)
        assert r["bleu_1"] > 0.9
        assert r["bleu_4"] > 0.9

    def test_bleu_empty(self):
        from src.evaluation.metrics import compute_bleu
        hyps = ["no match here at all"]
        refs = ["completely different text"]
        r = compute_bleu(hyps, refs)
        assert 0.0 <= r["bleu_4"] <= 1.0

    def test_rouge(self, sample_reports):
        from src.evaluation.metrics import compute_rouge
        hyps = sample_reports[:2]
        refs = sample_reports[:2]
        r = compute_rouge(hyps, refs)
        assert r["rouge_l"] > 0.9

    def test_radgraph_fallback(self, sample_reports):
        """Test the heuristic fallback (radgraph package may not be installed)."""
        from src.evaluation.metrics import _radgraph_heuristic_fallback
        r = _radgraph_heuristic_fallback(sample_reports[:2], sample_reports[:2])
        assert r["radgraph_f1"] is None
        assert 0.0 <= r["clinical_entity_overlap_f1"] <= 1.0

    def test_exact_match(self):
        from src.evaluation.metrics import compute_exact_match
        assert compute_exact_match(["yes"], ["yes"]) == 1.0
        assert compute_exact_match(["yes"], ["no"])  == 0.0
        assert compute_exact_match(["The Cat"], ["the cat"]) == 1.0

    def test_token_f1(self):
        from src.evaluation.metrics import compute_token_f1
        f1 = compute_token_f1(
            ["cardiomegaly with effusion"],
            ["mild cardiomegaly and pleural effusion"],
        )
        assert 0.0 < f1 < 1.0

    def test_recall_at_k(self):
        from src.evaluation.metrics import compute_recall_at_k
        retrieved = [["s1", "s2", "s3"], ["s4", "s5", "s6"]]
        relevant  = ["s1", "s5"]
        r1 = compute_recall_at_k(retrieved, relevant, k=1)
        r5 = compute_recall_at_k(retrieved, relevant, k=5)
        assert r1 == 0.5
        assert r5 == 1.0

    def test_mrr(self):
        from src.evaluation.metrics import compute_mrr
        retrieved = [["s1", "s2"], ["s3", "s1"]]
        relevant  = ["s1", "s1"]
        mrr = compute_mrr(retrieved, relevant)
        assert abs(mrr - 0.75) < 0.01  # (1.0 + 0.5) / 2


# ─────────────────────────────────────────────────────────────────────────────
#  Pipelines — Mock end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestReportGenerationPipeline:
    def test_run_mock(self, dummy_cxr):
        from src.pipelines.report_generation import ReportGenerationPipeline
        p = ReportGenerationPipeline(model="mock", mock=True).load()
        result = p.run(dummy_cxr)
        assert "findings" in result
        assert "impression" in result
        assert "latency_s" in result

    def test_run_batch_mock(self, dummy_cxr):
        from src.pipelines.report_generation import ReportGenerationPipeline
        p = ReportGenerationPipeline(model="mock", mock=True).load()
        samples = [{"image": dummy_cxr, "study_id": f"s{i}", "report": ""}
                   for i in range(3)]
        results = p.run_batch(samples)
        assert len(results) == 3
        assert all("findings" in r for r in results)


class TestRAGQAPipeline:
    def test_run_mock(self, dummy_cxr):
        from src.pipelines.rag_qa import RAGQAPipeline
        p = RAGQAPipeline(retriever="clip", mock=True).load()
        result = p.run(dummy_cxr, "Is there a pleural effusion?")
        assert "answer" in result
        assert "retrieved_studies" in result
        assert "latency_s" in result

    def test_build_index_and_run(self, dummy_cxr):
        from src.pipelines.rag_qa import RAGQAPipeline
        p = RAGQAPipeline(retriever="clip", mock=True).load()
        studies = [
            {"image": dummy_cxr, "report": "Normal CXR.", "study_id": f"s{i}"}
            for i in range(5)
        ]
        p.set_index(studies)
        result = p.run(dummy_cxr, "What is the impression?")
        assert len(result["answer"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
#  QA Dataset Generation
# ─────────────────────────────────────────────────────────────────────────────

class TestQADatasetGenerator:
    def test_parse_response_valid(self):
        from src.qa_dataset.generator import QADatasetGenerator
        gen = QADatasetGenerator()
        valid_json = json.dumps([
            {"question_type": "presence",
             "question": "Is there a pleural effusion?",
             "answer": "No.",
             "evidence": "No pleural effusion.",
             "difficulty": "easy"},
        ])
        result = gen._parse_response(valid_json)
        assert len(result) == 1
        assert result[0]["question"] == "Is there a pleural effusion?"

    def test_parse_response_with_fences(self):
        from src.qa_dataset.generator import QADatasetGenerator
        gen = QADatasetGenerator()
        fenced = '```json\n[{"question": "Q?", "answer": "A."}]\n```'
        result = gen._parse_response(fenced)
        assert len(result) == 1

    def test_filter_qa_pairs(self):
        from src.qa_dataset.generator import QADatasetGenerator
        pairs = [
            {"question": "Is there a pleural effusion?", "answer": "No effusion is identified.",
             "study_id": "s1"},
            {"question": "short?", "answer": "yes",  "study_id": "s2"},  # too short
            {"question": "What is the diagnosis?", "answer": "unknown",  "study_id": "s3"},  # degenerate
        ]
        filtered = QADatasetGenerator.filter_qa_pairs(pairs)
        assert len(filtered) == 1
        assert filtered[0]["study_id"] == "s1"

    def test_statistics(self):
        from src.qa_dataset.generator import QADatasetGenerator
        pairs = [
            {"question_type": "presence", "difficulty": "easy", "study_id": "s1",
             "question": "Q1?", "answer": "A1."},
            {"question_type": "location", "difficulty": "medium", "study_id": "s2",
             "question": "Q2?", "answer": "A2."},
        ]
        stats = QADatasetGenerator.get_statistics(pairs)
        assert stats["total_pairs"] == 2
        assert stats["by_question_type"]["presence"] == 1

    def test_save_load(self, tmp_path):
        from src.qa_dataset.generator import QADatasetGenerator
        pairs = [{"question": "Q?", "answer": "A.", "study_id": "s1",
                  "question_type": "presence"}]
        out = tmp_path / "qa.json"
        QADatasetGenerator.save(pairs, out)
        loaded = QADatasetGenerator.load(out)
        assert len(loaded) == 1
        assert loaded[0]["question"] == "Q?"


# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_load_default_config(self):
        from src.config import load_config
        cfg = load_config()
        if cfg:  # may be empty if running outside repo root
            assert "data" in cfg or "models" in cfg

    def test_get_device(self):
        from src.config import get_device
        d = get_device()
        assert d in ("cuda", "mps", "cpu")

    def test_is_mock_mode(self):
        from src.config import is_mock_mode
        assert is_mock_mode() is True  # MOCK_MODELS=true set at top


# ─────────────────────────────────────────────────────────────────────────────
#  Prompts
# ─────────────────────────────────────────────────────────────────────────────

class TestPrompts:
    def test_get_report_prompt_default(self):
        from src.models.prompts import get_report_prompt
        p = get_report_prompt()
        assert "FINDINGS" in p
        assert "IMPRESSION" in p

    def test_get_qa_prompt_rag(self):
        from src.models.prompts import get_qa_prompt
        p = get_qa_prompt("rag", question="Is there effusion?", context="No effusion.")
        assert "Is there effusion?" in p
        assert "No effusion." in p

    def test_report_prompt_registry(self):
        from src.models.prompts import REPORT_PROMPTS, QA_PROMPTS
        assert "medgemma_report_v1" in REPORT_PROMPTS
        assert "rag" in QA_PROMPTS


# ─────────────────────────────────────────────────────────────────────────────
#  Integration: full pipeline (mock)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_report_gen_then_evaluate(self, dummy_cxr):
        from src.pipelines.report_generation import ReportGenerationPipeline
        from src.evaluation.metrics import compute_bleu, compute_rouge

        pipeline = ReportGenerationPipeline(model="mock", mock=True).load()
        sample = {"image": dummy_cxr, "report": "FINDINGS:\nNormal.\n\nIMPRESSION:\nNo acute process."}
        results = pipeline.run_batch([sample])

        generated = [r.get("raw", r.get("findings", "")) for r in results]
        references = [sample["report"]]

        bleu = compute_bleu(generated, references)
        rouge = compute_rouge(generated, references)
        assert "bleu_4" in bleu
        assert "rouge_l" in rouge

    def test_rag_qa_end_to_end(self, dummy_cxr):
        from src.pipelines.rag_qa import RAGQAPipeline
        from src.evaluation.qa_evaluator import QAEvaluator

        pipeline = RAGQAPipeline(retriever="clip", mock=True).load()
        result = pipeline.run(dummy_cxr, "Is there cardiomegaly?")

        evaluator = QAEvaluator()
        metrics = evaluator.evaluate(
            predictions=[result["answer"]],
            references=["No cardiomegaly is identified."],
            questions=["Is there cardiomegaly?"],
        )
        assert "exact_match" in metrics
        assert "token_f1" in metrics
        assert 0.0 <= metrics["token_f1"] <= 1.0
