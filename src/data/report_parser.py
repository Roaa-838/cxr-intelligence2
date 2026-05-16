"""src/data/report_parser.py — Parse radiology report free text into sections."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReportSections:
    """Structured representation of a radiology report."""
    raw: str
    findings: str = ""
    impression: str = ""
    technique: str = ""
    comparison: str = ""
    has_findings: bool = False
    has_impression: bool = False

    @property
    def full_text(self) -> str:
        """Canonical formatted report."""
        parts: list[str] = []
        if self.technique:
            parts.append(f"TECHNIQUE:\n{self.technique.strip()}")
        if self.comparison:
            parts.append(f"COMPARISON:\n{self.comparison.strip()}")
        if self.findings:
            parts.append(f"FINDINGS:\n{self.findings.strip()}")
        if self.impression:
            parts.append(f"IMPRESSION:\n{self.impression.strip()}")
        return "\n\n".join(parts) if parts else self.raw


class ReportParser:
    """
    Parse free-text radiology reports into structured sections.

    MIMIC-CXR reports typically contain:
        TECHNIQUE (optional)
        COMPARISON (optional)
        FINDINGS   ← main descriptive section
        IMPRESSION ← clinical summary / conclusion

    Handles common formatting variations:
        - All-caps section headers: FINDINGS:, IMPRESSION:
        - Mixed case: Findings:, Impression:
        - With/without trailing colon
        - Sections in different orders
        - Reports with only one section (common in short CXR reads)
    """

    # Section header patterns (case-insensitive)
    _SECTION_HEADERS = {
        "technique":   r"\b(TECHNIQUE|CLINICAL\s+INFORMATION|CLINICAL\s+INDICATION|INDICATION)\s*:?",
        "comparison":  r"\b(COMPARISON|PRIOR\s+STUDY|PRIOR\s+FILMS?)\s*:?",
        "findings":    r"\b(FINDINGS?|RADIOGRAPHIC\s+FINDINGS?)\s*:?",
        "impression":  r"\b(IMPRESSION|CONCLUSION|ASSESSMENT|INTERPRETATION|SUMMARY)\s*:?",
    }

    # Ordered list for sectioning
    _SECTION_ORDER = ["technique", "comparison", "findings", "impression"]

    def __init__(self, strict: bool = False) -> None:
        """
        Args:
            strict: If True, return empty strings for sections not found.
                    If False (default), assign unlabelled text to findings.
        """
        self.strict = strict
        # Pre-compile patterns
        self._patterns = {
            name: re.compile(pat, re.IGNORECASE)
            for name, pat in self._SECTION_HEADERS.items()
        }

    def parse(self, report_text: str) -> ReportSections:
        """
        Parse a raw report string into structured sections.

        Args:
            report_text: Raw free-text radiology report.

        Returns:
            ReportSections dataclass.
        """
        text = report_text.strip()
        if not text:
            return ReportSections(raw=text)

        # Find all section header positions
        boundaries: list[tuple[int, str]] = []
        for section_name, pattern in self._patterns.items():
            for m in pattern.finditer(text):
                boundaries.append((m.end(), section_name))

        # If no section headers found
        if not boundaries:
            return self._handle_unsectioned(text)

        # Sort by position in text
        boundaries.sort(key=lambda x: x[0])

        sections: dict[str, str] = {s: "" for s in self._SECTION_ORDER}

        for i, (start, sec_name) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            # Remove the matched header from content
            content = text[start:end].strip()
            # Strip the header text that might bleed in
            for other_pattern in self._patterns.values():
                content = other_pattern.sub("", content, count=1).strip()
            if sec_name in sections:
                sections[sec_name] = content

        # Capture any text before the first header (likely unlabelled findings)
        preamble_end = boundaries[0][0] - len(
            re.search(list(self._patterns.values())[0].pattern,
                      text[:boundaries[0][0]] or "", re.IGNORECASE
            ).group(0)
            if re.search(list(self._patterns.values())[0].pattern,
                         text[:boundaries[0][0]], re.IGNORECASE)
            else ""
        )
        preamble = text[:preamble_end].strip() if preamble_end > 0 else ""
        if preamble and not self.strict and not sections["findings"]:
            sections["findings"] = preamble

        result = ReportSections(
            raw=text,
            findings=self._clean(sections["findings"]),
            impression=self._clean(sections["impression"]),
            technique=self._clean(sections["technique"]),
            comparison=self._clean(sections["comparison"]),
            has_findings=bool(sections["findings"].strip()),
            has_impression=bool(sections["impression"].strip()),
        )
        return result

    def _handle_unsectioned(self, text: str) -> ReportSections:
        """
        Handle reports without explicit section headers.
        Heuristic: last 1-2 sentences → impression; rest → findings.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if len(sentences) <= 1:
            return ReportSections(
                raw=text,
                findings="" if self.strict else text,
                impression=text,
                has_findings=not self.strict,
                has_impression=True,
            )

        # Assign last sentence as impression, rest as findings
        impression = sentences[-1]
        findings = " ".join(sentences[:-1])
        return ReportSections(
            raw=text,
            findings=findings,
            impression=impression,
            has_findings=True,
            has_impression=True,
        )

    @staticmethod
    def _clean(text: str) -> str:
        """Remove extra whitespace and artefacts."""
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ── Convenience wrappers ─────────────────────────────────────────────────

    def get_findings(self, report_text: str) -> str:
        return self.parse(report_text).findings

    def get_impression(self, report_text: str) -> str:
        return self.parse(report_text).impression

    def get_sections_dict(self, report_text: str) -> dict[str, str]:
        s = self.parse(report_text)
        return {"findings": s.findings, "impression": s.impression,
                "technique": s.technique, "comparison": s.comparison}


# ── Clinical entity extraction (lightweight, no ML needed) ────────────────────

# Patterns for common CXR findings (used in QA dataset creation)
FINDING_PATTERNS = {
    "pneumothorax":    r"\b(pneumothorax|pnx)\b",
    "pleural_effusion":r"\b(pleural\s+effusion|hydrothorax)\b",
    "consolidation":   r"\b(consolidat(?:ion|ed))\b",
    "atelectasis":     r"\b(atelectasis|atelectatic)\b",
    "cardiomegaly":    r"\b(cardiomegaly|enlarged\s+heart|cardiac\s+enlargement)\b",
    "edema":           r"\b(edema|oedema|pulmonary\s+congestion|vascular\s+congestion)\b",
    "pneumonia":       r"\b(pneumonia|infectious)\b",
    "fracture":        r"\b(fracture|rib\s+fracture)\b",
    "nodule":          r"\b(nodule|mass|opacity)\b",
    "no_finding":      r"\b(no\s+acute|unremarkable|within\s+normal\s+limits|clear)\b",
}


def extract_findings_presence(text: str) -> dict[str, bool]:
    """
    Lightweight regex-based clinical finding extraction.
    
    Returns a dict mapping finding name → True if mentioned in text.
    Note: does NOT distinguish positive from negated statements.
    Use NLP NER (NegEx, scispaCy) for that.
    """
    text_lower = text.lower()
    return {
        name: bool(re.search(pattern, text_lower, re.IGNORECASE))
        for name, pattern in FINDING_PATTERNS.items()
    }
