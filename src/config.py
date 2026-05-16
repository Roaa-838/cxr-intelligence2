"""src/config.py — Central configuration loader."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load YAML config, merge with environment overrides.

    Args:
        config_path: Path to a YAML config file. Defaults to configs/default.yaml
                     relative to the project root.

    Returns:
        Merged configuration dictionary.
    """
    if config_path is None:
        # Walk up to find configs/default.yaml from anywhere inside the project
        here = Path(__file__).resolve().parent
        for parent in [here, here.parent, here.parent.parent]:
            candidate = parent / "configs" / "default.yaml"
            if candidate.exists():
                config_path = candidate
                break
        else:
            logger.warning("configs/default.yaml not found; using empty config.")
            return {}

    with open(config_path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    # ── Env overrides ────────────────────────────────────────────────────────
    _apply_env_overrides(cfg)
    return cfg


def _apply_env_overrides(cfg: dict) -> None:
    """Mutate cfg in-place with environment variable overrides."""
    env_map = {
        "HF_TOKEN":            ("models", "hf_token"),
        "MIMIC_CXR_PATH":      ("data", "mimic_cxr_path"),
        "PROCESSED_DATA_PATH": ("data", "processed_path"),
        "INDEX_PATH":          ("rag", "index_path"),
        "RESULTS_PATH":        ("logging", "results_dir"),
        "MEDGEMMA_MODEL_ID":   ("models", "medgemma", "model_id"),
        "COLPALI_MODEL_ID":    ("models", "colpali", "model_id"),
        "DEVICE":              ("device",),
        "MOCK_MODELS":         ("mock_models",),
        "USE_4BIT":            ("models", "medgemma", "use_4bit"),
    }

    for env_key, cfg_path in env_map.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        # Convert booleans
        if val.lower() in ("true", "1", "yes"):
            val = True
        elif val.lower() in ("false", "0", "no"):
            val = False
        # Drill into nested dict
        d = cfg
        for part in cfg_path[:-1]:
            d = d.setdefault(part, {})
        d[cfg_path[-1]] = val


def get_device(cfg: dict | None = None) -> str:
    """Return the best available device string."""
    import torch
    if cfg and cfg.get("device"):
        return cfg["device"]
    env_device = os.environ.get("DEVICE", "").lower()
    if env_device in ("cuda", "mps", "cpu"):
        return env_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_mock_mode(cfg: dict | None = None) -> bool:
    """Return True if running without real model weights (CI / CPU testing)."""
    if cfg and cfg.get("mock_models"):
        return bool(cfg["mock_models"])
    return os.environ.get("MOCK_MODELS", "false").lower() in ("true", "1", "yes")
