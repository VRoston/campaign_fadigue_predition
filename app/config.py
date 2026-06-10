"""Configuration loader for the Streamlit app."""

import os
from pathlib import Path
from typing import Optional


def get_data_path() -> Path:
    """Get the raw data directory from environment or default."""
    env = os.environ.get("CAMPAIGN_DATA_PATH")
    if env:
        return Path(env).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "data" / "raw"


def get_model_path() -> Optional[Path]:
    """Get the model path from environment."""
    env = os.environ.get("CAMPAIGN_MODEL_PATH")
    if env:
        return Path(env).expanduser().resolve()

    # Try default location: models/v4_lagged/lgbm_fatigue_v4.joblib
    repo_root = Path(__file__).resolve().parents[1]
    default_model = repo_root.parent / "models" / "v4_lagged" / "lgbm_fatigue_v4.joblib"
    if default_model.exists():
        return default_model

    return None


def get_model_summary_path() -> Optional[Path]:
    """Get the model summary path from environment or default."""
    env = os.environ.get("CAMPAIGN_MODEL_ROOT")
    if env:
        return Path(env).expanduser().resolve() / "v4_lagged" / "model_summary.json"

    # Try default location: models/v4_lagged/model_summary.json
    repo_root = Path(__file__).resolve().parents[1]
    default_summary = repo_root.parent / "models" / "v4_lagged" / "model_summary.json"
    if default_summary.exists():
        return default_summary

    return None


def get_alert_threshold() -> float:
    """Get the alert threshold from environment or default (0.328 for fatigue)."""
    env = os.environ.get("ALERT_THRESHOLD", "0.328")
    try:
        return float(env)
    except ValueError:
        return 0.328
