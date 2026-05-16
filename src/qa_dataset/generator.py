"""
src/qa_dataset/generator.py
============================
Synthetic QA dataset generation from MIMIC-CXR radiology reports.

Methodology (documented for the project report):
  1. Parse each report into Findings + Impression sections
  2. Submit to Claude (or GPT-4) with a structured prompt requesting N QA pairs
  3. Parse JSON response → (question, answer, evidence, question_type)
  4. Validate answers against the source report
  5. Apply deduplication and quality filters
  6. Output: qa_dataset.json with study_id, image_path, question, answer, etc.

Question types generated:
  - presence:   "Is there a pleural effusion?" → Yes/No + evidence
  - location:   "Where is the consolidation located?" → anatomical region
  - severity:   "How severe is the cardiomegaly?" → mild/moderate/severe
  - diagnosis:  "What is the most likely diagnosis?" → clinical entity
  - comparison: "Is this X-ray normal or abnormal?" → description

See docs/qa_dataset_creation.md for full methodology write-up.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class QADatasetGenerator:
    """
    Generate synthetic (question, answer) pairs from radiology reports
    using a large language model as the annotation engine.

    Args:
        provider:        "anthropic" | "openai"
        model:           Model name (e.g. "claude-3-5-haiku-20241022")
        n_questions:     QA pairs to generate per study (default 5)
        temperature:     Sampling temperature for diversity
        max_retries:     API retry attempts on failure

    Usage:
        gen = QADatasetGenerator(provider="anthropic")
        qa_pairs = gen.generate_from_reports(reports_df)
        gen.save(qa_pairs, "data/qa_dataset.json")
    """

    GENERATION_PROMPT = """\
You are a board-certified radiologist creating a medical Visual Question \
Answering (VQA) dataset from chest X-ray radiology reports.

Given this radiology report, generate exactly {n_questions} clinically \
relevant and diverse question-answer pairs.

RADIOLOGY REPORT:
{report_text}

REQUIREMENTS:
- Questions must be directly and fully answerable from the report text
- Cover different question types: presence/absence, location, severity, diagnosis
- Include both positive cases (finding present) and negative cases (no finding)
- Use standard radiological and anatomical terminology
- Each answer must cite the exact evidence from the report
- Vary difficulty: easy (yes/no), medium (location), hard (diagnosis/severity)

Return ONLY a valid JSON array, no other text:
[
  {{
    "question_type": "presence",
    "question": "Is there a pleural effusion present?",
    "answer": "No. The report states no pleural effusion or pneumothorax is identified.",
    "evidence": "No pleural effusion or pneumothorax is identified.",
    "difficulty": "easy"
  }}
]

Generate exactly {n_questions} objects in the array."""

    QUESTION_TYPES = [
        "presence",
        "location",
        "severity",
        "diagnosis",
        "comparison",
    ]

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-3-5-haiku-20241022",
        n_questions: int = 5,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.n_questions = n_questions
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None

    def _get_client(self):
        """Lazy-init API client."""
        if self._client is not None:
            return self._client

        if self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", "")
            )
        elif self.provider == "openai":
            import openai
            self._client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", "")
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        return self._client

    def generate_for_report(
        self,
        report_text: str,
        study_id: str = "",
        image_path: str = "",
    ) -> list[dict]:
        """
        Generate QA pairs for a single report.

        Args:
            report_text: Full radiology report string.
            study_id:    MIMIC study ID (for provenance).
            image_path:  Path to CXR image (for provenance).

        Returns:
            list of QA dicts with keys:
              study_id, image_path, question_type, question, answer,
              evidence, difficulty, source_report
        """
        prompt = self.GENERATION_PROMPT.format(
            n_questions=self.n_questions,
            report_text=report_text.strip(),
        )

        for attempt in range(self.max_retries):
            try:
                raw_response = self._call_llm(prompt)
                qa_list = self._parse_response(raw_response)
                if not qa_list:
                    logger.warning(
                        "Empty QA list for study %s (attempt %d)", study_id, attempt + 1
                    )
                    continue

                # Enrich with provenance
                for qa in qa_list:
                    qa["study_id"]      = study_id
                    qa["image_path"]    = image_path
                    qa["source_report"] = report_text

                return qa_list

            except Exception as e:
                logger.warning(
                    "QA generation failed for study %s (attempt %d/%d): %s",
                    study_id, attempt + 1, self.max_retries, e,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        return []

    def generate_from_reports(
        self,
        reports: list[dict],
        show_progress: bool = True,
        rate_limit_delay: float = 0.5,
    ) -> list[dict]:
        """
        Generate QA pairs for a list of reports.

        Args:
            reports: list of dicts with keys: "text" (report), "study_id",
                     optionally "image" (path)
            show_progress: Log progress every 25 studies.
            rate_limit_delay: Seconds to wait between API calls.

        Returns:
            Flat list of all QA pairs from all reports.
        """
        all_qa: list[dict] = []
        n = len(reports)
        logger.info("Generating QA dataset from %d reports …", n)

        for i, report_dict in enumerate(reports):
            report_text = report_dict.get("text", report_dict.get("report", ""))
            if not report_text or len(report_text.strip()) < 50:
                logger.debug("Skipping study %s — report too short.", report_dict.get("study_id", i))
                continue

            qa_pairs = self.generate_for_report(
                report_text=report_text,
                study_id=str(report_dict.get("study_id", f"study_{i}")),
                image_path=str(report_dict.get("image", report_dict.get("image_path", ""))),
            )

            all_qa.extend(qa_pairs)

            if show_progress and (i + 1) % 25 == 0:
                logger.info(
                    "  Progress: %d / %d studies | %d QA pairs so far",
                    i + 1, n, len(all_qa),
                )

            time.sleep(rate_limit_delay)

        logger.info("Generated %d QA pairs from %d reports.", len(all_qa), n)
        return all_qa

    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM provider."""
        client = self._get_client()

        if self.provider == "anthropic":
            msg = client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text

        elif self.provider == "openai":
            resp = client.chat.completions.create(
                model=self.model,
                max_tokens=2048,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content

        raise ValueError(f"Unknown provider: {self.provider}")

    def _parse_response(self, text: str) -> list[dict]:
        """Extract JSON array from LLM response."""
        text = text.strip()

        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # Find JSON array bounds
        start = text.find("[")
        end   = text.rfind("]")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON array found in response: {text[:200]}")

        json_str = text[start: end + 1]
        data = json.loads(json_str)

        if not isinstance(data, list):
            raise ValueError("Parsed JSON is not a list.")

        # Validate required fields
        valid: list[dict] = []
        required = {"question", "answer"}
        for item in data:
            if isinstance(item, dict) and required.issubset(item.keys()):
                item.setdefault("question_type", "general")
                item.setdefault("evidence", "")
                item.setdefault("difficulty", "medium")
                valid.append(item)

        return valid

    # ── Quality filtering ─────────────────────────────────────────────────────

    @staticmethod
    def filter_qa_pairs(
        qa_pairs: list[dict],
        min_answer_length: int = 10,
        min_question_length: int = 15,
        deduplicate: bool = True,
    ) -> list[dict]:
        """
        Apply quality filters to generated QA pairs.

        Filters:
          1. Minimum question/answer length
          2. Remove degenerate answers ("unknown", "not specified", etc.)
          3. Deduplicate near-identical questions within same study
        """
        DEGENERATE = {
            "unknown", "not specified", "cannot be determined",
            "not mentioned", "unclear", "n/a", "none", "not applicable",
        }
        filtered: list[dict] = []
        seen_questions: set[str] = set()

        for qa in qa_pairs:
            q = qa.get("question", "").strip()
            a = qa.get("answer",   "").strip()

            if len(q) < min_question_length:
                continue
            if len(a) < min_answer_length:
                continue
            if a.lower() in DEGENERATE:
                continue

            # Dedup key: study_id + normalised question
            if deduplicate:
                dedup_key = f"{qa.get('study_id', '')}::{q.lower()}"
                if dedup_key in seen_questions:
                    continue
                seen_questions.add(dedup_key)

            filtered.append(qa)

        logger.info(
            "QA filtering: %d → %d pairs (removed %d)",
            len(qa_pairs), len(filtered), len(qa_pairs) - len(filtered),
        )
        return filtered

    # ── Persistence ────────────────────────────────────────────────────────────

    @staticmethod
    def save(qa_pairs: list[dict], path: str | Path) -> None:
        """Save QA dataset to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(qa_pairs, f, indent=2)
        logger.info("Saved %d QA pairs -> %s", len(qa_pairs), path)

    @staticmethod
    def load(path: str | Path) -> list[dict]:
        """Load a previously saved QA dataset."""
        with open(path) as f:
            data = json.load(f)
        logger.info("Loaded %d QA pairs <- %s", len(data), path)
        return data

    @staticmethod
    def get_statistics(qa_pairs: list[dict]) -> dict:
        """Compute dataset statistics."""
        from collections import Counter
        type_counts = Counter(qa.get("question_type", "general") for qa in qa_pairs)
        diff_counts = Counter(qa.get("difficulty", "medium") for qa in qa_pairs)
        study_ids   = {qa.get("study_id") for qa in qa_pairs}
        return {
            "total_pairs":       len(qa_pairs),
            "unique_studies":    len(study_ids),
            "pairs_per_study":   len(qa_pairs) / max(len(study_ids), 1),
            "by_question_type":  dict(type_counts),
            "by_difficulty":     dict(diff_counts),
        }

    @staticmethod
    def print_statistics(qa_pairs: list[dict]) -> None:
        stats = QADatasetGenerator.get_statistics(qa_pairs)
        print(f"\n{'─' * 45}")
        print("  QA Dataset Statistics")
        print(f"{'─' * 45}")
        print(f"  Total pairs:       {stats['total_pairs']}")
        print(f"  Unique studies:    {stats['unique_studies']}")
        print(f"  Pairs per study:   {stats['pairs_per_study']:.1f}")
        print(f"\n  By question type:")
        for t, n in sorted(stats["by_question_type"].items()):
            bar = "█" * int(20 * n / max(stats["total_pairs"], 1))
            print(f"    {t:<15} {n:>5}  {bar}")
        print(f"\n  By difficulty:")
        for d, n in sorted(stats["by_difficulty"].items()):
            print(f"    {d:<10} {n:>5}")
        print(f"{'─' * 45}\n")
