from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import json
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
    # Determine expected features (by name) if possible
    expected_features: list[str] | None = None
    if feature_columns:
        expected_features = list(feature_columns)
    else:
        # 1) sklearn / xgboost wrapper attribute
        try:
            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
        except Exception:
            expected_features = None

        # 2) xgboost booster feature names
        if not expected_features:
            try:
                booster_get = getattr(model, "get_booster", None)
                if callable(booster_get):
                    b = model.get_booster()
                    fn = getattr(b, "feature_names", None)
                    if fn:
                        expected_features = list(fn)
            except Exception:
                expected_features = None

        # 3) embedded model_summary on the object
        if not expected_features:
            try:
                summary = getattr(model, "model_summary", None)
                if isinstance(summary, dict) and summary.get("features_used"):
                    expected_features = list(summary.get("features_used"))
            except Exception:
                expected_features = None

        # 4) fallback: try model_summary.json in common locations
        if not expected_features:
            try:
                repo_root = Path(__file__).resolve().parents[2]
                try_paths = []
                env_root = os.environ.get("CAMPAIGN_MODEL_ROOT")
                if env_root:
                    try_paths.append(Path(env_root) / "model_summary.json")
                try_paths.append(repo_root / "app" / "models" / "model_summary.json")
                for sp in try_paths:
                    if sp.exists():
                        with open(sp, "r", encoding="utf8") as fh:
                            ms = json.load(fh)
                        if isinstance(ms, dict) and ms.get("features_used"):
                            expected_features = list(ms.get("features_used"))
                            break
            except Exception:
                expected_features = None

    # Build features DataFrame aligned to expected features when possible
    if expected_features:
        features = enriched.copy()
        for feat in expected_features:
            if feat not in features.columns:
                features[feat] = 0.0
        # keep only expected features in order
        features = features.reindex(columns=expected_features)
    else:
        # fallback: numeric columns except the output
        numeric_cols = enriched.select_dtypes(include=["number"]).columns.tolist()
        if output_column in numeric_cols:
            numeric_cols.remove(output_column)
        features = enriched[numeric_cols].copy()

    # coerce numeric and fill
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Align by count if model exposes expected number of features
    model_n = None
    try:
        model_n = int(getattr(model, "n_features_in_", None)) if getattr(model, "n_features_in_", None) is not None else None
    except Exception:
        model_n = None

    if model_n is None:
        try:
            booster_get = getattr(model, "get_booster", None)
            if callable(booster_get):
                b = model.get_booster()
                fn = getattr(b, "feature_names", None)
                if fn:
                    model_n = len(fn)
                else:
                    numf = getattr(b, "num_features", None)
                    if callable(numf):
                        try:
                            model_n = int(b.num_features())
                        except Exception:
                            model_n = None
        except Exception:
            model_n = None

    warnings: list[str] = []
    cur_n = features.shape[1]
    if model_n is not None and cur_n != model_n:
        if cur_n < model_n:
            pad_count = model_n - cur_n
            for i in range(pad_count):
                features[f"_pad_{i}"] = 0.0
            warnings.append(f"Added {pad_count} zero-padding columns to match model expected features ({model_n}).")
        else:
            features = features.iloc[:, :model_n]
            warnings.append(f"Trimmed input features from {cur_n} to {model_n} to match model expected features.")

    try:
        preds = model.predict(features)
        enriched[output_column] = preds
        # attach warnings to DataFrame attrs for callers that inspect them
        if warnings:
            enriched.attrs["prediction_warnings"] = warnings
    except Exception as exc:
        enriched[output_column] = np.nan
        enriched.attrs["prediction_error"] = str(exc)
        if warnings:
            enriched.attrs["prediction_warnings"] = warnings

    return enriched


def get_model_info() -> dict:
    """Return a small metadata dictionary about the loaded model.

    The function attempts to extract commonly used fields that a serialized
    model might carry (e.g. `model_summary`, `feature_importances_`, `coef_`).
    It always returns a dict with safe defaults so the UI can rely on keys.
    """
    info = {
        "has_model": False,
        "model_version": None,
        "alert_threshold": None,
        "metrics_holdout_auc": None,
        "last_trained": None,
        "feature_importances": None,
    }
    try:
        model = load_fatigue_model()
    except Exception:
        # no model available via default loader — try to read a model_summary.json
        repo_root = Path(__file__).resolve().parents[2]
        # check CAMPAIGN_MODEL_ROOT first
        env_root = os.environ.get("CAMPAIGN_MODEL_ROOT")
        summary_paths = []
        if env_root:
            summary_paths.append(Path(env_root) / "model_summary.json")
        summary_paths.append(repo_root / "app" / "models" / "model_summary.json")
        for sp in summary_paths:
            try:
                if sp.exists():
                    with open(sp, "r", encoding="utf8") as fh:
                        ms = json.load(fh)
                    info["model_version"] = ms.get("version") or ms.get("model_version")
                    info["alert_threshold"] = ms.get("alert_threshold")
                    info["metrics_holdout_auc"] = ms.get("metrics_holdout_auc")
                    info["last_trained"] = ms.get("last_trained")
                    return info
            except Exception:
                continue

        # fallback default
        info["alert_threshold"] = float(os.environ.get("ALERT_THRESHOLD", 0.5))
        return info

    info["has_model"] = True

    # some model objects may embed a 'model_summary' dict
    summary = getattr(model, "model_summary", None)
    if isinstance(summary, dict):
        info["model_version"] = summary.get("model_version")
        info["alert_threshold"] = summary.get("alert_threshold")
        info["metrics_holdout_auc"] = summary.get("metrics_holdout_auc")
        info["last_trained"] = summary.get("last_trained")
    else:
        # try reading model_summary.json near the models folder
        try:
            repo_root = Path(__file__).resolve().parents[2]
            summ_file = repo_root / "app" / "models" / "model_summary.json"
            if summ_file.exists():
                import json as _json

                ms = _json.load(open(summ_file, "r", encoding="utf8"))
                info["model_version"] = info.get("model_version") or ms.get("version") or ms.get("model_version")
                info["alert_threshold"] = info.get("alert_threshold") or ms.get("alert_threshold")
                info["metrics_holdout_auc"] = info.get("metrics_holdout_auc") or ms.get("metrics_holdout_auc")
        except Exception:
            pass

    # feature importances for tree-based models
    fi = getattr(model, "feature_importances_", None)
    if fi is not None:
        try:
            info["feature_importances"] = list(map(float, fi))
        except Exception:
            info["feature_importances"] = None

    # feature names if available
    try:
        fn = getattr(model, "feature_names_in_", None)
        if fn is not None:
            info["feature_names"] = list(fn)
        else:
            # try xgboost booster
            booster_get = getattr(model, "get_booster", None)
            if callable(booster_get):
                try:
                    b = model.get_booster()
                    fn2 = getattr(b, "feature_names", None)
                    if fn2:
                        info["feature_names"] = list(fn2)
                except Exception:
                    pass
    except Exception:
        pass

    # coefficients for linear models
    coef = getattr(model, "coef_", None)
    if coef is not None and info["feature_importances"] is None:
        try:
            # flatten and take absolute as importance proxy
            import numpy as _np

            arr = _np.asarray(coef)
            info["feature_importances"] = list(map(float, _np.abs(arr).ravel()))
        except Exception:
            pass

    # fallback default threshold
    if info["alert_threshold"] is None:
        info["alert_threshold"] = float(os.environ.get("ALERT_THRESHOLD", 0.5))

    return info
