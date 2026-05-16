"""src/models/prompts.py — Prompt templates for MedGemma and other VLMs."""

from __future__ import annotations

from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  REPORT GENERATION PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

REPORT_PROMPT_V1 = """\
You are an expert radiologist. Carefully examine this chest X-ray image and \
generate a complete, structured radiology report.

Your report MUST include exactly two sections:

FINDINGS:
Systematically describe all visible structures. Cover:
1. Heart and mediastinum (size, borders, contour)
2. Lungs (bilateral assessment, any opacities, consolidation, hyperinflation)
3. Pleura (effusion, pneumothorax, thickening)
4. Bones (ribs, spine, clavicles — fractures, lytic lesions)
5. Soft tissues and diaphragm
6. Any tubes, lines, or devices if present

IMPRESSION:
Provide a concise clinical summary of key findings and diagnoses. \
List the most clinically significant finding first. \
If normal, state: "No acute cardiopulmonary process."

Be specific, accurate, and use standard radiological terminology. \
Do NOT fabricate findings not visible in the image."""

REPORT_PROMPT_V2_DETAILED = """\
You are a board-certified radiologist performing a systematic chest X-ray \
interpretation. Your task is to generate a detailed, clinically accurate \
radiology report.

INSTRUCTIONS:
- Use the ABCDE systematic approach: Airways, Bones, Cardiac, Diaphragm, \
Everything else
- Note laterality (left/right) for all asymmetric findings
- Quantify where possible (e.g., "small", "moderate", "large" effusion; \
"mild", "moderate", "severe" cardiomegaly)
- Use standard anatomical terms (e.g., "right lower lobe", \
"left costophrenic angle")

FINDINGS:
[Your systematic findings here]

IMPRESSION:
[Your clinical summary here]"""

REPORT_PROMPT_RAG = """\
You are an expert radiologist. You have access to similar previous cases for \
reference.

RETRIEVED SIMILAR CASES:
{context}

Based on the chest X-ray image and the context from similar cases above, \
generate a complete radiology report. Use the retrieved cases only as \
reference — base your report primarily on what you observe in THIS image.

FINDINGS:
[Describe all findings in this chest X-ray]

IMPRESSION:
[Clinical summary and conclusions]"""


# ─────────────────────────────────────────────────────────────────────────────
#  QA PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

QA_PROMPT_ZERO_SHOT = """\
You are an expert radiologist answering a clinical question about a \
chest X-ray.

Question: {question}

Based on your analysis of this chest X-ray image, provide a specific, \
accurate, and clinically relevant answer. Be concise but complete."""

QA_PROMPT_RAG = """\
You are an expert radiologist answering a clinical question about a \
chest X-ray. You have retrieved similar cases from the knowledge base \
for reference.

QUESTION: {question}

RETRIEVED SIMILAR CASES FOR CONTEXT:
{context}

INSTRUCTIONS:
- Answer the question based on what you observe in THIS chest X-ray image
- Use the retrieved cases only as clinical context/reference
- Be specific (mention anatomical location, severity, laterality)
- If the answer requires clinical correlation, state so
- Do NOT fabricate findings; if a finding cannot be determined, say so

ANSWER:"""

QA_PROMPT_STRUCTURED = """\
You are a radiologist answering questions about chest X-rays.

Question: {question}
Question type: {question_type}

Context from similar cases:
{context}

Provide your answer in this format:
ANSWER: [direct answer to the question]
CONFIDENCE: [High / Medium / Low]
REASONING: [brief explanation of your reasoning]"""


# ─────────────────────────────────────────────────────────────────────────────
#  QA DATASET GENERATION PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

QA_GENERATION_PROMPT = """\
You are a board-certified radiologist creating a medical visual question \
answering (VQA) dataset from chest X-ray reports.

Given the following radiology report, generate {n_questions} clinically \
relevant question-answer pairs. The questions should be answerable from \
the report and cover different aspects of the study.

RADIOLOGY REPORT:
{report_text}

QUESTION TYPES TO COVER:
- presence: Is [finding] present? (Yes/No with brief justification)
- location: Where is [finding] located? (anatomical region)
- severity: How severe is [finding]? (mild/moderate/severe)
- diagnosis: What is the most likely diagnosis?
- comparison: Is this normal or abnormal? What is abnormal?

IMPORTANT:
- Questions must be directly answerable from the report
- Include both positive (finding present) and negative (finding absent) cases
- Use standard medical/radiological terminology
- Each answer must cite the relevant part of the report

Return ONLY a valid JSON array with no additional text:
[
  {{
    "question_type": "presence",
    "question": "Is there a pleural effusion?",
    "answer": "No. The report states no pleural effusion is identified.",
    "evidence": "No pleural effusion or pneumothorax is identified.",
    "difficulty": "easy"
  }},
  ...
]"""


# ─────────────────────────────────────────────────────────────────────────────
#  LLM-AS-JUDGE PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

LLM_JUDGE_REPORT_PROMPT = """\
You are an expert radiologist evaluating the quality of AI-generated \
chest X-ray reports.

REFERENCE REPORT (ground truth):
{reference}

GENERATED REPORT:
{generated}

Rate the generated report on each dimension (1-5 scale):

1. CLINICAL ACCURACY: Are the findings correct? No hallucinations?
   (5=fully correct, 4=mostly correct, 3=mixed, 2=mostly wrong, 1=incorrect)

2. COMPLETENESS: Are all important findings mentioned?
   (5=fully complete, 4=minor omissions, 3=moderate omissions, 2=major omissions, 1=incomplete)

3. SPECIFICITY: Is the report specific (location, laterality, severity)?
   (5=very specific, 4=mostly specific, 3=moderate, 2=vague, 1=non-specific)

4. CLINICAL UTILITY: Would this report guide appropriate patient management?
   (5=definitely, 4=probably, 3=possibly, 2=unlikely, 1=no)

Return ONLY valid JSON:
{{
  "clinical_accuracy": <1-5>,
  "completeness": <1-5>,
  "specificity": <1-5>,
  "clinical_utility": <1-5>,
  "overall": <1-5>,
  "comments": "<brief explanation>"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

REPORT_PROMPTS = {
    "medgemma_report_v1":          REPORT_PROMPT_V1,
    "medgemma_report_v2_detailed": REPORT_PROMPT_V2_DETAILED,
    "medgemma_report_rag":         REPORT_PROMPT_RAG,
}

QA_PROMPTS = {
    "zero_shot":  QA_PROMPT_ZERO_SHOT,
    "rag":        QA_PROMPT_RAG,
    "structured": QA_PROMPT_STRUCTURED,
}


def get_report_prompt(name: str = "medgemma_report_v1", **kwargs) -> str:
    """Get a report generation prompt by name, formatting any placeholders."""
    template = REPORT_PROMPTS.get(name, REPORT_PROMPT_V1)
    return template.format(**kwargs) if kwargs else template


def get_qa_prompt(name: str = "rag", **kwargs) -> str:
    """Get a QA prompt by name, filling in placeholders."""
    template = QA_PROMPTS.get(name, QA_PROMPT_RAG)
    return template.format(**kwargs) if kwargs else template
