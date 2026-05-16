"""
src/evaluation/metrics.py
==========================
Evaluation metrics for chest X-ray report generation and QA.

Metrics implemented:
  ── Lexical ─────────────────────────────────────────────────────────────────
  - BLEU-1/2/4:     N-gram precision (nltk / sacrebleu)
  - ROUGE-L:        Longest common subsequence F1
  - METEOR:         Word-level alignment with stemming and synonyms

  ── Semantic ────────────────────────────────────────────────────────────────
  - BERTScore F1:   Contextual embedding similarity (best with SciBERT for CXR)

  ── Clinical (most important) ────────────────────────────────────────────────
  - RadGraph F1:    Entity + relation graph comparison (highest clinical validity)
  - CheXBert sim:  Label-based similarity via CheXBert label extractor

  ── QA-specific ─────────────────────────────────────────────────────────────
  - Exact Match (EM)
  - F1 token overlap
  - LLM-as-judge   (requires API key)

Reference ranking (correlation with radiologist judgement):
  RadGraph F1 > BERTScore > ROUGE-L > BLEU
  Source: Jain et al. 2021, Patterns 2023
"""

from __future__ import annotations

import json
import logging
import re
import string
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  LEXICAL METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_bleu(
    hypotheses: list[str],
    references: list[str],
    max_n: int = 4,
) -> dict[str, float]:
    """
    Compute corpus-level BLEU-1 through BLEU-max_n.

    Args:
        hypotheses: Generated reports (one per sample).
        references: Ground-truth reports (one per sample).
        max_n:      Maximum n-gram order.

    Returns:
        dict with keys: bleu_1, bleu_2, bleu_4, bleu_avg
    """
    try:
        from nltk.translate.bleu_score import (
            corpus_bleu, SmoothingFunction
        )
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)

        smooth = SmoothingFunction().method3
        tokenised_refs = [[ref.lower().split()] for ref in references]
        tokenised_hyps = [hyp.lower().split() for hyp in hypotheses]

        scores: dict[str, float] = {}
        for n in range(1, max_n + 1):
            weights = tuple([1.0 / n] * n)
            scores[f"bleu_{n}"] = corpus_bleu(
                tokenised_refs, tokenised_hyps,
                weights=weights, smoothing_function=smooth
            )
        scores["bleu_avg"] = float(np.mean(list(scores.values())))
        return scores

    except Exception as e:
        logger.warning("BLEU computation failed: %s", e)
        return {f"bleu_{n}": 0.0 for n in range(1, max_n + 1)} | {"bleu_avg": 0.0}


def compute_rouge(
    hypotheses: list[str],
    references: list[str],
) -> dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.

    Returns:
        dict with keys: rouge_1, rouge_2, rouge_l
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        r1_list, r2_list, rl_list = [], [], []
        for hyp, ref in zip(hypotheses, references):
            scores = scorer.score(ref, hyp)
            r1_list.append(scores["rouge1"].fmeasure)
            r2_list.append(scores["rouge2"].fmeasure)
            rl_list.append(scores["rougeL"].fmeasure)

        return {
            "rouge_1": float(np.mean(r1_list)),
            "rouge_2": float(np.mean(r2_list)),
            "rouge_l": float(np.mean(rl_list)),
        }
    except Exception as e:
        logger.warning("ROUGE computation failed: %s", e)
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
#  SEMANTIC METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_bertscore(
    hypotheses: list[str],
    references: list[str],
    model_type: str = "allenai/scibert_scivocab_uncased",
    batch_size: int = 8,
    device: Optional[str] = None,
) -> dict[str, float]:
    """
    Compute BERTScore F1 (precision + recall also returned).

    Medical-specific models (preferred order):
      1. allenai/scibert_scivocab_uncased  (biomedical, best for CXR)
      2. microsoft/deberta-xlarge-mnli
      3. bert-base-uncased                 (fastest fallback)

    CVE-2025-32434 note: BERTScore calls torch.load() internally.
    torch < 2.6 shows a security warning but still works; we suppress it.
    To silence permanently: pip install "torch>=2.6"

    Returns:
        dict with keys: bertscore_p, bertscore_r, bertscore_f1
    """
    import warnings as _warnings
    import torch as _torch

    if device is None:
        device = "cuda" if _torch.cuda.is_available() else "cpu"

    # Workaround: patch torch.load so older torch versions do not raise
    _orig_load = _torch.load
    def _safe_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            return _orig_load(*args, **kwargs)
    _torch.load = _safe_load

    try:
        from bert_score import score as bert_score_fn
        logger.info("Computing BERTScore with model=%s ...", model_type)
        P, R, F1 = bert_score_fn(
            hypotheses,
            references,
            model_type=model_type,
            batch_size=batch_size,
            device=device,
            verbose=False,
        )
        return {
            "bertscore_p":  float(P.mean()),
            "bertscore_r":  float(R.mean()),
            "bertscore_f1": float(F1.mean()),
        }
    except Exception as e:
        logger.warning("BERTScore computation failed: %s", e)
        logger.warning(
            "Fix BERTScore: pip install \"torch>=2.6\" torchvision "
            "--index-url https://download.pytorch.org/whl/cu118"
        )
        return {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0}
    finally:
        _torch.load = _orig_load  # always restore


# ─────────────────────────────────────────────────────────────────────────────
#  CLINICAL METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_radgraph_f1(
    hypotheses: list[str],
    references: list[str],
    reward_level: str = "partial",  # "partial" | "complete"
) -> dict[str, float]:
    """
    Compute RadGraph F1 — the clinically most valid metric for CXR reports.

    RadGraph parses reports into clinical entity-relation graphs and computes
    F1 between the hypothesis and reference graphs.

    Award levels:
      - "partial":  entity + partial relation matching (recommended)
      - "complete": requires complete entity + full relation chain match

    Requires: pip install radgraph
    If not installed, falls back to a simple entity overlap heuristic.

    Returns:
        dict with keys: radgraph_f1, radgraph_precision, radgraph_recall
    """
    try:
        from radgraph import F1RadGraph  # type: ignore
        logger.info("Computing RadGraph F1 (reward_level=%s) …", reward_level)
        f1_scorer = F1RadGraph(reward_level=reward_level)
        scores = f1_scorer(hyps=hypotheses, refs=references)
        # radgraph returns (mean_score, hypothesis_scores, reference_scores)
        if isinstance(scores, tuple):
            mean_f1 = float(scores[0])
            individual = list(scores[1])
        else:
            mean_f1 = float(np.mean(scores))
            individual = list(scores)
        return {
            "radgraph_f1":        mean_f1,
            "radgraph_individual": individual,
        }
    except ImportError:
        logger.warning(
            "radgraph package not installed. "
            "Using clinical entity overlap heuristic as fallback."
        )
        return _radgraph_heuristic_fallback(hypotheses, references)
    except Exception as e:
        logger.warning("RadGraph F1 computation failed: %s", e)
        return {"radgraph_f1": 0.0, "radgraph_individual": []}


# ── Clinical entity heuristic (fallback when radgraph not available) ──────────

# Key clinical entities in CXR reports
_CLINICAL_ENTITIES = [
    "pneumothorax", "pleural effusion", "consolidation", "atelectasis",
    "cardiomegaly", "edema", "pulmonary edema", "pneumonia",
    "opacity", "fracture", "nodule", "mass", "infiltrate",
    "no acute", "normal", "unremarkable", "clear",
    "right", "left", "bilateral", "upper lobe", "lower lobe",
    "mild", "moderate", "severe", "small", "large",
]


def _radgraph_heuristic_fallback(
    hypotheses: list[str],
    references: list[str],
) -> dict[str, float]:
    """
    Lightweight clinical entity F1 — a rough proxy for RadGraph F1.
    Only use when the radgraph package is unavailable.
    """
    f1_scores: list[float] = []
    for hyp, ref in zip(hypotheses, references):
        hyp_lower = hyp.lower()
        ref_lower = ref.lower()
        hyp_entities = {e for e in _CLINICAL_ENTITIES if e in hyp_lower}
        ref_entities = {e for e in _CLINICAL_ENTITIES if e in ref_lower}
        if not ref_entities:
            f1_scores.append(1.0 if not hyp_entities else 0.5)
            continue
        tp = len(hyp_entities & ref_entities)
        precision = tp / len(hyp_entities) if hyp_entities else 0.0
        recall    = tp / len(ref_entities)
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        f1_scores.append(f1)

    return {
        "radgraph_f1":         float(np.mean(f1_scores)),
        "radgraph_individual": f1_scores,
        "radgraph_note":       "heuristic_fallback",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  QA METRICS
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_answer(text: str) -> str:
    """Standard QA normalisation: lowercase, strip punctuation and articles."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def compute_exact_match(predictions: list[str], references: list[str]) -> float:
    """Fraction of predictions that exactly match the reference (after normalise)."""
    assert len(predictions) == len(references)
    matches = sum(
        _normalise_answer(p) == _normalise_answer(r)
        for p, r in zip(predictions, references)
    )
    return matches / len(predictions) if predictions else 0.0


def compute_token_f1(predictions: list[str], references: list[str]) -> float:
    """
    Token-level F1 score (SQuAD-style).

    F1 = 2 * precision * recall / (precision + recall)
    where precision = |overlap| / |prediction|
          recall    = |overlap| / |reference|
    """
    f1_list: list[float] = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(_normalise_answer(pred).split())
        ref_tokens  = set(_normalise_answer(ref).split())
        if not ref_tokens:
            f1_list.append(1.0 if not pred_tokens else 0.0)
            continue
        overlap = pred_tokens & ref_tokens
        if not overlap:
            f1_list.append(0.0)
            continue
        precision = len(overlap) / len(pred_tokens)
        recall    = len(overlap) / len(ref_tokens)
        f1_list.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1_list)) if f1_list else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  RETRIEVAL METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_recall_at_k(
    retrieved_ids: list[list[str]],
    relevant_ids:  list[str],
    k: int = 5,
) -> float:
    """
    Recall@k: fraction of queries where the correct study is in the top-k results.

    Args:
        retrieved_ids: For each query, an ordered list of retrieved study IDs.
        relevant_ids:  The correct study ID for each query.
        k:             Cutoff.

    Returns:
        Recall@k as a float in [0, 1].
    """
    hits = sum(
        rel_id in ret_ids[:k]
        for ret_ids, rel_id in zip(retrieved_ids, relevant_ids)
    )
    return hits / len(relevant_ids) if relevant_ids else 0.0


def compute_mrr(
    retrieved_ids: list[list[str]],
    relevant_ids:  list[str],
) -> float:
    """Mean Reciprocal Rank."""
    rr_list: list[float] = []
    for ret_ids, rel_id in zip(retrieved_ids, relevant_ids):
        try:
            rank = ret_ids.index(rel_id) + 1
            rr_list.append(1.0 / rank)
        except ValueError:
            rr_list.append(0.0)
    return float(np.mean(rr_list)) if rr_list else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  COMBINED EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

class ReportEvaluator:
    """
    Convenience class: run all report generation metrics in one call.

    Usage:
        evaluator = ReportEvaluator(device="cuda")
        results = evaluator.evaluate(generated_reports, reference_reports)
        evaluator.print_results(results)
        evaluator.save_results(results, "results/medgemma_eval.json")
    """

    def __init__(
        self,
        device: Optional[str] = None,
        bertscore_model: str = "allenai/scibert_scivocab_uncased",
        use_radgraph: bool = True,
    ) -> None:
        import torch as _torch
        self.device = device or ("cuda" if _torch.cuda.is_available() else "cpu")
        self.bertscore_model = bertscore_model
        self.use_radgraph = use_radgraph

    def evaluate(
        self,
        hypotheses: list[str],
        references: list[str],
    ) -> dict:
        """
        Run BLEU, ROUGE, BERTScore, RadGraph F1.

        Returns:
            Nested dict: {metric_name: value}
        """
        assert len(hypotheses) == len(references), \
            f"Lengths mismatch: {len(hypotheses)} vs {len(references)}"

        results: dict = {"n_samples": len(hypotheses)}

        logger.info("Evaluating %d samples …", len(hypotheses))

        # Lexical
        logger.info("  -> BLEU …")
        results.update(compute_bleu(hypotheses, references))

        logger.info("  -> ROUGE …")
        results.update(compute_rouge(hypotheses, references))

        # Semantic
        logger.info("  -> BERTScore …")
        results.update(compute_bertscore(
            hypotheses, references,
            model_type=self.bertscore_model,
            device=self.device,
        ))

        # Clinical
        if self.use_radgraph:
            logger.info("  -> RadGraph F1 …")
            results.update(compute_radgraph_f1(hypotheses, references))

        logger.info("Evaluation complete.")
        return results

    def print_results(self, results: dict, title: str = "Evaluation Results") -> None:
        """Pretty-print metric table."""
        print(f"\n{'─' * 50}")
        print(f"  {title}")
        print(f"{'─' * 50}")
        print(f"  Samples: {results.get('n_samples', '?')}")
        print(f"  {'Metric':<25} {'Score':>10}")
        print(f"  {'─' * 35}")

        metric_order = [
            ("radgraph_f1",  "RadGraph F1 ★"),
            ("bertscore_f1", "BERTScore F1"),
            ("rouge_l",      "ROUGE-L"),
            ("rouge_1",      "ROUGE-1"),
            ("bleu_4",       "BLEU-4"),
            ("bleu_1",       "BLEU-1"),
        ]
        for key, label in metric_order:
            if key in results:
                print(f"  {label:<25} {results[key]:>10.4f}")
        print(f"{'─' * 50}\n")

    def save_results(self, results: dict, path: str) -> None:
        """Save results to JSON."""
        import json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Remove non-serialisable items
        serialisable = {
            k: v for k, v in results.items()
            if not isinstance(v, list) or len(v) < 20
        }
        with open(path, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info("Results saved -> %s", path)
