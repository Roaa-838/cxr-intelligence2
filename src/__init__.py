"""
CXR Intelligence System
=======================
Multi-Modal Chest X-Ray Report Generation & QA System

Modes:
  1. Report Generation  — image → structured radiology report
  2. RAG-Based QA       — image + question → grounded clinical answer

Models: MedGemma-4B-it (primary), ColPali-v1.2 (retrieval), BiomedCLIP (comparison)
"""

__version__ = "1.0.0"
__author__ = "CXR Intelligence Team"
