"""
src/evaluation/qa_evaluator.py
================================
QA evaluation: Exact Match, Token F1, BERTScore, and LLM-as-judge.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .metrics import (
    compute_exact_match, compute_token_f1,
    compute_bertscore, compute_rouge,
)

logger = logging.getLogger(__name__)


class QAEvaluator:
    """
    Evaluate generated QA answers against reference answers.

    Metrics:
      - Exact Match (EM): exact string match after normalisation
      - Token F1: token-level overlap (SQuAD-style)
      - BERTScore F1: semantic similarity via SciBERT
      - ROUGE-L: longest common subsequence F1
      - LLM-as-judge (optional): uses Claude / GPT-4 as clinical judge

    Usage:
        evaluator = QAEvaluator()
        results = evaluator.evaluate(
            predictions=["No pleural effusion."],
            references=["No pleural effusion is identified."],
            questions=["Is there a pleural effusion?"],
        )
    """

    LLM_JUDGE_PROMPT = """\
You are an expert radiologist evaluating AI-generated answers to chest X-ray
clinical questions.

Question: {question}
Reference Answer: {reference}
Generated Answer: {generated}

Rate the generated answer on a scale of 0-5:
5 = Clinically accurate, complete, well-explained
4 = Clinically correct with minor omissions
3 = Mostly correct but missing key clinical details
2 = Partially correct, significant errors present
1 = Mostly incorrect but some relevant content
0 = Completely wrong or contains hallucinated findings

Return ONLY a JSON object:
{{"score": <0-5>, "reasoning": "<one sentence>"}}"""

    def __init__(
        self,
        device: Optional[str] = None,
        bertscore_model: str = "allenai/scibert_scivocab_uncased",
        use_llm_judge: bool = False,
        llm_judge_provider: str = "anthropic",
        llm_judge_model: str = "claude-3-5-haiku-20241022",
    ) -> None:
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bertscore_model = bertscore_model
        self.use_llm_judge = use_llm_judge
        self.llm_judge_provider = llm_judge_provider
        self.llm_judge_model = llm_judge_model

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
        questions: Optional[list[str]] = None,
        question_types: Optional[list[str]] = None,
        max_judge_samples: int = 50,
    ) -> dict:
        """
        Run all QA metrics.

        Args:
            predictions:     Generated answers.
            references:      Ground-truth answers.
            questions:       Original questions (needed for LLM judge).
            question_types:  "presence" | "location" | "severity" | "diagnosis" etc.
            max_judge_samples: Max samples to send to LLM judge (costs money).

        Returns:
            dict with all metric scores + per-sample breakdown.
        """
        n = len(predictions)
        assert n == len(references), f"Length mismatch: {n} vs {len(references)}"
        logger.info("QA evaluation over %d samples …", n)

        results: dict = {"n_samples": n}

        # ── Exact Match ──────────────────────────────────────────────────────
        results["exact_match"] = compute_exact_match(predictions, references)

        # ── Token F1 ─────────────────────────────────────────────────────────
        results["token_f1"] = compute_token_f1(predictions, references)

        # ── ROUGE-L ──────────────────────────────────────────────────────────
        rouge = compute_rouge(predictions, references)
        results["rouge_l"] = rouge["rouge_l"]
        results["rouge_1"] = rouge["rouge_1"]

        # ── BERTScore ────────────────────────────────────────────────────────
        bs = compute_bertscore(
            predictions, references,
            model_type=self.bertscore_model,
            device=self.device,
        )
        results["bertscore_f1"] = bs["bertscore_f1"]
        results["bertscore_p"]  = bs["bertscore_p"]
        results["bertscore_r"]  = bs["bertscore_r"]

        # ── Per-type breakdown ───────────────────────────────────────────────
        if question_types:
            results["by_type"] = self._compute_by_type(
                predictions, references, question_types
            )

        # ── LLM-as-judge ─────────────────────────────────────────────────────
        if self.use_llm_judge and questions:
            logger.info("Running LLM-as-judge on up to %d samples …", max_judge_samples)
            judge_preds = predictions[:max_judge_samples]
            judge_refs  = references[:max_judge_samples]
            judge_qs    = questions[:max_judge_samples]
            judge_scores = self._run_llm_judge(judge_preds, judge_refs, judge_qs)
            results["llm_judge_mean"] = float(np.mean(judge_scores))
            results["llm_judge_scores"] = judge_scores

        return results

    def _compute_by_type(
        self,
        predictions: list[str],
        references:  list[str],
        question_types: list[str],
    ) -> dict:
        """Break down EM and token F1 by question type."""
        from collections import defaultdict
        type_preds: dict[str, list] = defaultdict(list)
        type_refs:  dict[str, list] = defaultdict(list)
        for p, r, t in zip(predictions, references, question_types):
            type_preds[t].append(p)
            type_refs[t].append(r)

        by_type: dict = {}
        for qtype in sorted(type_preds.keys()):
            p_list = type_preds[qtype]
            r_list = type_refs[qtype]
            by_type[qtype] = {
                "n":           len(p_list),
                "exact_match": compute_exact_match(p_list, r_list),
                "token_f1":    compute_token_f1(p_list, r_list),
            }
        return by_type

    def _run_llm_judge(
        self,
        predictions: list[str],
        references:  list[str],
        questions:   list[str],
    ) -> list[float]:
        """Call Claude / OpenAI to judge answer quality."""
        scores: list[float] = []

        for pred, ref, q in zip(predictions, references, questions):
            prompt = self.LLM_JUDGE_PROMPT.format(
                question=q, reference=ref, generated=pred
            )
            score = self._call_llm_judge(prompt)
            scores.append(score)

        return scores

    def _call_llm_judge(self, prompt: str) -> float:
        """Call the LLM API and parse the numeric score."""
        try:
            if self.llm_judge_provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY", "")
                )
                msg = client.messages.create(
                    model=self.llm_judge_model,
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = msg.content[0].text.strip()
            elif self.llm_judge_provider == "openai":
                import openai
                client = openai.OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY", "")
                )
                resp = client.chat.completions.create(
                    model=self.llm_judge_model,
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.choices[0].message.content.strip()
            else:
                return 0.0

            data = json.loads(text)
            return float(data.get("score", 0))

        except Exception as e:
            logger.warning("LLM judge call failed: %s", e)
            return 0.0

    def print_results(self, results: dict, title: str = "QA Evaluation") -> None:
        print(f"\n{'─' * 50}")
        print(f"  {title}")
        print(f"{'─' * 50}")
        print(f"  Samples:       {results.get('n_samples', '?')}")
        print(f"  {'Metric':<25} {'Score':>10}")
        print(f"  {'─' * 35}")
        ordered = [
            ("bertscore_f1", "BERTScore F1"),
            ("token_f1",     "Token F1"),
            ("rouge_l",      "ROUGE-L"),
            ("exact_match",  "Exact Match"),
            ("llm_judge_mean", "LLM Judge (0-5)"),
        ]
        for key, label in ordered:
            if key in results:
                val = results[key]
                # LLM judge is 0-5, others 0-1
                fmt = f"{val:>10.4f}" if key != "llm_judge_mean" else f"{val:>10.2f}/5.0"
                print(f"  {label:<25} {fmt}")

        if "by_type" in results:
            print(f"\n  {'─' * 45}")
            print(f"  {'By Question Type':<25}")
            print(f"  {'Type':<20} {'N':>5} {'EM':>8} {'F1':>8}")
            for qtype, stats in results["by_type"].items():
                print(f"  {qtype:<20} {stats['n']:>5} "
                      f"{stats['exact_match']:>8.3f} {stats['token_f1']:>8.3f}")
        print(f"{'─' * 50}\n")

    def save_results(self, results: dict, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("QA results saved -> %s", path)
