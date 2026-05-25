from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _default_model_path() -> Path:
    env = os.environ.get("CAMPAIGN_MODEL_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).with_name("modelo_fadiga.joblib")


@lru_cache(maxsize=1)
def load_fatigue_model(model_path: Path | str | None = None):
    path = Path(model_path) if model_path else _default_model_path()
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Model file {path} not found or empty.")
    return joblib.load(path)


def enrich_with_fatigue_prediction(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    output_column: str = "fadiga_prevista",
) -> pd.DataFrame:
    enriched = df.copy()

    if enriched.empty:
        enriched[output_column] = pd.Series(dtype=float)
        return enriched

    try:
        model = load_fatigue_model()
    except Exception:
        enriched[output_column] = np.nan
        return enriched

    if feature_columns is None:
        if hasattr(model, "feature_names_in_"):
            feature_columns = [
                col for col in model.feature_names_in_ if col in enriched.columns
            ]
        else:
            numeric_cols = enriched.select_dtypes(include=["number"]).columns.tolist()
            feature_columns = [col for col in numeric_cols if col != output_column]

    if not feature_columns:
        enriched[output_column] = np.nan
        return enriched

    features = enriched[feature_columns].copy()
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    predictions = model.predict(features)
    enriched[output_column] = predictions
    return enriched
