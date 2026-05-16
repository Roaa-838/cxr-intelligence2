#!/usr/bin/env python3
"""
scripts/generate_qa_dataset.py
================================
Generate a synthetic QA dataset from MIMIC-CXR test-set reports.

This is a key original contribution: since MIMIC-CXR has no official QA
annotations, we use an LLM (Claude or GPT-4) to generate clinically valid
question-answer pairs from the free-text reports.

See docs/qa_dataset_creation.md for full methodology.

Usage:
    # Generate from real data with Claude
    python scripts/generate_qa_dataset.py \
        --data-dir data/processed \
        --output data/qa_dataset.json \
        --n-questions 5 \
        --split test \
        --provider anthropic

    # Quick test (synthetic data, no API key needed)
    python scripts/generate_qa_dataset.py --synthetic --n-samples 10

    # Validate and show stats on existing QA dataset
    python scripts/generate_qa_dataset.py --validate data/qa_dataset.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def generate_mock_qa(reports: list[dict], n_questions: int) -> list[dict]:
    """
    Generate mock QA pairs without any API (for testing pipeline).
    Uses template-based generation from parsed report sections.
    """
    from src.data.report_parser import ReportParser, extract_findings_presence

    parser = ReportParser()
    qa_pairs: list[dict] = []

    QUESTION_TEMPLATES = {
        "pneumothorax":     ("Is there a pneumothorax?", "pneumothorax"),
        "pleural_effusion": ("Is there a pleural effusion?", "pleural effusion"),
        "consolidation":    ("Is there consolidation in the lungs?", "consolidation"),
        "atelectasis":      ("Is atelectasis present?", "atelectasis"),
        "cardiomegaly":     ("Is cardiomegaly present?", "cardiomegaly"),
        "edema":            ("Is there pulmonary edema?", "pulmonary edema"),
        "no_finding":       ("Are the lungs clear?", "lungs"),
    }

    for report_dict in reports:
        report = report_dict.get("text", report_dict.get("report", ""))
        if not report:
            continue

        sections = parser.parse(report)
        findings = extract_findings_presence(sections.findings or report)
        study_id = str(report_dict.get("study_id", ""))
        img_path = str(report_dict.get("image_path", report_dict.get("image", "")))

        # Generate presence/absence questions
        for finding_key, (question, term) in QUESTION_TEMPLATES.items():
            present = findings.get(finding_key, False)
            answer = (
                f"Yes, {term} is mentioned in the report. "
                f"See findings: {sections.findings[:150]}…"
                if present else
                f"No, there is no mention of {term} in this report."
            )
            qa_pairs.append({
                "study_id":      study_id,
                "image_path":    img_path,
                "question_type": "presence",
                "question":      question,
                "answer":        answer,
                "evidence":      sections.findings[:200],
                "difficulty":    "easy",
                "source_report": report,
                "generated_by":  "template_mock",
            })
            if len(qa_pairs) % n_questions == 0:
                break  # n_questions per study

        # Add diagnosis question from impression
        if sections.impression:
            qa_pairs.append({
                "study_id":      study_id,
                "image_path":    img_path,
                "question_type": "diagnosis",
                "question":      "What is the radiological impression / primary diagnosis?",
                "answer":        sections.impression,
                "evidence":      sections.impression,
                "difficulty":    "medium",
                "source_report": report,
                "generated_by":  "template_mock",
            })

    return qa_pairs


def parse_args():
    p = argparse.ArgumentParser(description="Generate QA dataset from CXR reports")
    p.add_argument("--data-dir",    default="data/processed",
                   help="Processed data directory with metadata.csv")
    p.add_argument("--output",      default="data/qa_dataset.json",
                   help="Output path for QA dataset JSON")
    p.add_argument("--n-samples",   type=int, default=500,
                   help="Number of studies to generate QA from")
    p.add_argument("--n-questions", type=int, default=5,
                   help="QA pairs per study")
    p.add_argument("--split",       default="test",
                   help="Dataset split to use")
    p.add_argument("--provider",    default="anthropic",
                   choices=["anthropic", "openai", "mock"],
                   help="LLM provider. Use 'mock' for testing without API keys.")
    p.add_argument("--model",       default="claude-3-5-haiku-20241022",
                   help="LLM model name")
    p.add_argument("--synthetic",   action="store_true",
                   help="Use synthetic data (no real dataset needed)")
    p.add_argument("--validate",    default=None,
                   help="Path to existing QA JSON to validate (no generation)")
    p.add_argument("--filter",      action="store_true",
                   help="Apply quality filters before saving")
    return p.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from src.qa_dataset.generator import QADatasetGenerator

    # ── Validate mode ─────────────────────────────────────────────────────────
    if args.validate:
        logger.info("Validating QA dataset: %s", args.validate)
        qa_pairs = QADatasetGenerator.load(args.validate)
        QADatasetGenerator.print_statistics(qa_pairs)
        return

    # ── Load reports ──────────────────────────────────────────────────────────
    if args.synthetic:
        from src.data.dataset import SyntheticCXRDataset
        n = min(args.n_samples, 200)
        ds = SyntheticCXRDataset(n_samples=n)
        reports = [
            {"study_id": ds[i].study_id, "text": ds[i].report,
             "image_path": ds[i].image_path}
            for i in range(len(ds))
        ]
        logger.info("Using %d synthetic reports.", len(reports))
    else:
        csv = Path(args.data_dir) / "metadata.csv"
        if not csv.exists():
            logger.error("metadata.csv not found at %s. Run download_data.py first.", csv)
            sys.exit(1)
        df = pd.read_csv(csv)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if "split" in df.columns:
            df = df[df["split"] == args.split]
        df = df.dropna(subset=["text"]).head(args.n_samples)
        reports = df[["study_id", "text"] + (["image"] if "image" in df.columns else [])].to_dict("records")
        logger.info("Loaded %d reports from %s split.", len(reports), args.split)

    # ── Generate QA ───────────────────────────────────────────────────────────
    if args.provider == "mock" or args.synthetic:
        logger.info("Using mock/template QA generation (no API call) …")
        qa_pairs = generate_mock_qa(reports, args.n_questions)
    else:
        logger.info("Generating QA with %s/%s …", args.provider, args.model)
        gen = QADatasetGenerator(
            provider=args.provider,
            model=args.model,
            n_questions=args.n_questions,
        )
        qa_pairs = gen.generate_from_reports(reports)

    # ── Quality filter ─────────────────────────────────────────────────────────
    if args.filter:
        qa_pairs = QADatasetGenerator.filter_qa_pairs(qa_pairs)

    # ── Save and report ───────────────────────────────────────────────────────
    QADatasetGenerator.save(qa_pairs, output_path)
    QADatasetGenerator.print_statistics(qa_pairs)
    logger.info("Done. QA dataset -> %s", output_path)


if __name__ == "__main__":
    main()
