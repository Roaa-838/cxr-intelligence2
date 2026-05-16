"""
tests/conftest.py
==================
Shared pytest fixtures and environment setup.
Sets MOCK_MODELS=true globally so no GPU is needed for any test.
"""
import os
import sys
from pathlib import Path

# Always run tests in mock mode — no GPU required
os.environ["MOCK_MODELS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Make src importable from tests/
sys.path.insert(0, str(Path(__file__).parent.parent))
