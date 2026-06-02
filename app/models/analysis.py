from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from app.models.predictor import load_fatigue_model
import re


MODEL_ROOT_ENV = "CAMPAIGN_MODEL_ROOT"
RESULT_ROOT_ENV = "CAMPAIGN_RESULT_PATH"
CAMPAIGN_MODEL_PATH_ENV = "CAMPAIGN_MODEL_PATH"


def find_latest_analysis_file(campaign_id: str, result_root: Optional[str] = None) -> Optional[str]:
    """Find the latest analysis artifact (parquet or csv) for a campaign in the result root.

    Returns the path to the latest file or None when not found.
    """
    # resolve result root
    if result_root:
        root = Path(result_root)
    else:
        env_root = os.environ.get(RESULT_ROOT_ENV)
        if env_root:
            root = Path(env_root)
        else:
            root = Path.home() / "data" / "result_campaigns"

    if not root.exists() or not root.is_dir():
        return None

    # prefer a fixed filename per campaign (overwrite behavior). If not present,
    # fall back to timestamped artifacts for backward compatibility.
    fixed_parquet = root / f"{campaign_id}_analysis.parquet"
    fixed_csv = root / f"{campaign_id}_analysis.csv"
    if fixed_parquet.exists():
        return str(fixed_parquet)
    # prefer timestamped parquet files if any
    candidates = list(root.glob(f"{campaign_id}_analysis_*.parquet"))
    if candidates:
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0])
    # fall back to fixed csv
    if fixed_csv.exists():
        return str(fixed_csv)
    # finally timestamped csv
    candidates = list(root.glob(f"{campaign_id}_analysis_*.csv"))
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def find_campaign_model(campaign_id: str, model_root: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Locate model artifacts for a given campaign.

    The function looks for the following layout (in order):
      1) <model_root>/<campaign_id>/xgb_alert_model.joblib
      2) <model_root>/<campaign_id>.joblib
      3) CAMPAIGN_MODEL_PATH env var (single joblib)

    Returns dict with keys: model_path, encoders_path, summary_path, model_dir
    Values are strings or None when not found.
    """
    # resolve roots
    repo_root = Path(__file__).resolve().parents[2]
    app_models_dir = repo_root / "app" / "models"

    if model_root:
        root = Path(model_root)
    else:
        env_root = os.environ.get(MODEL_ROOT_ENV)
        if env_root:
            root = Path(env_root)
        elif app_models_dir.exists():
            # prefer model artifacts bundled under app/models when present
            root = app_models_dir
        else:
            root = repo_root / "data" / "models"

    # candidate dir
    candidate_dir = root / str(campaign_id)
    model_path = None
    encoders_path = None
    summary_path = None

    if candidate_dir.is_dir():
        m = candidate_dir / "xgb_alert_model.joblib"
        if m.exists():
            model_path = str(m)
        else:
            # try any .joblib in dir
            for p in candidate_dir.glob("*.joblib"):
                model_path = str(p)
                break

        enc = candidate_dir / "label_encoders.joblib"
        if enc.exists():
            encoders_path = str(enc)

        summ = candidate_dir / "model_summary.json"
        if summ.exists():
            summary_path = str(summ)


    # fallback: file named after campaign in root
    if not model_path:
        alt = root / f"{campaign_id}.joblib"
        if alt.exists():
            model_path = str(alt)

    # additional fallback: single model file at root (e.g. app/models/xgb_alert_model.joblib)
    if not model_path:
        common = root / "xgb_alert_model.joblib"
        if common.exists():
            model_path = str(common)

    # also look for a top-level model_summary.json at the root
    if not summary_path:
        root_summary = root / "model_summary.json"
        if root_summary.exists():
            summary_path = str(root_summary)

    # also look for a top-level label_encoders.joblib at the root
    if not encoders_path:
        root_enc = root / "label_encoders.joblib"
        if root_enc.exists():
            encoders_path = str(root_enc)

    # final fallback: single path env var
    if not model_path and os.environ.get(CAMPAIGN_MODEL_PATH_ENV):
        p = Path(os.environ.get(CAMPAIGN_MODEL_PATH_ENV))
        if p.exists():
            model_path = str(p)

    model_dir = None
    if candidate_dir.exists():
        model_dir = str(candidate_dir)
    elif model_path:
        model_dir = str(Path(model_path).parent)

    return {
        "model_path": model_path,
        "encoders_path": encoders_path,
        "summary_path": summary_path,
        "model_dir": model_dir,
    }


def run_campaign_analysis(
    df: pd.DataFrame,
    campaign_id: str,
    model_root: Optional[str] = None,
    result_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Run analysis for a single campaign and save results to disk.

    Saves a CSV with per-snapshot predictions and a JSON summary. Returns a dict
    with paths or an 'error' key on failure.
    """
    if df.empty:
        return {"error": "Dataframe vazio — nada para analisar."}

    locate = find_campaign_model(campaign_id, model_root=model_root)
    model_path = locate.get("model_path")

    if not model_path:
        return {
            "error": (
                "Modelo não encontrado. Coloque os artefatos em '<MODEL_ROOT>/<campaign_id>/' "
                "(xgb_alert_model.joblib, label_encoders.joblib, model_summary.json) ou defina CAMPAIGN_MODEL_PATH."
            )
        }

    try:
        model = load_fatigue_model(model_path)
    except Exception as exc:
        return {"error": f"Falha ao carregar o modelo: {exc}"}

    # Try to load model_summary (if present) to get canonical feature list and threshold
    summary_path = locate.get("summary_path")
    model_summary = None
    if summary_path:
        try:
            with open(summary_path, "r", encoding="utf8") as fh:
                model_summary = json.load(fh)
        except Exception:
            model_summary = None

    # Determine expected features (in order)
    expected_features = None
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    elif model_summary and isinstance(model_summary, dict) and model_summary.get("features_used"):
        expected_features = list(model_summary.get("features_used"))
    else:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        expected_features = [c for c in numeric_cols if c != "fadiga_prevista"]

    if not expected_features:
        return {"error": "Nenhuma coluna de features disponível para o modelo."}

    # Build a features DataFrame by mapping expected feature names to available columns.
    # Normalization removes non-alphanum and lowercases for fuzzy matching.
    def _norm(s: str) -> str:
        return re.sub(r"[^0-9a-z]", "", str(s).lower())

    col_norm = {c: _norm(c) for c in df.columns}
    features = pd.DataFrame(index=df.index)
    for feat in expected_features:
        nfeat = _norm(feat)
        found = None
        # exact or suffix match preferred
        for c, nc in col_norm.items():
            if nc == nfeat or nc.endswith(nfeat) or nfeat in nc:
                found = c
                break

        if found:
            features[feat] = df[found]
        else:
            # fallback: zero-filled column
            features[feat] = 0.0

    # Try to apply label encoders if available (best-effort)
    enc_path = locate.get("encoders_path")
    if enc_path:
        try:
            encoders = joblib.load(enc_path)
            if isinstance(encoders, dict):
                for col, enc in encoders.items():
                    if col in features.columns:
                        try:
                            features[col] = enc.transform(features[col].astype(str).fillna(""))
                        except Exception:
                            features[col] = pd.to_numeric(features[col], errors="coerce")
        except Exception:
            encoders = None

    # Ensure numeric matrix (coerce any remaining values)
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Attempt to detect model's expected feature count and adapt if necessary
    warnings: list[str] = []
    model_n = None
    try:
        model_n = int(getattr(model, "n_features_in_", None)) if getattr(model, "n_features_in_", None) is not None else None
    except Exception:
        model_n = None

    if model_n is None:
        # try to inspect xgboost booster feature names
        try:
            booster = getattr(model, "get_booster", None)
            if callable(booster):
                b = model.get_booster()
                fn = getattr(b, "feature_names", None)
                if fn:
                    model_n = len(fn)
                else:
                    # some xgboost versions expose num_features()
                    numf = getattr(b, "num_features", None)
                    if callable(numf):
                        try:
                            model_n = int(b.num_features())
                        except Exception:
                            model_n = None
        except Exception:
            model_n = None

    cur_n = features.shape[1]
    if model_n is not None and cur_n != model_n:
        if cur_n < model_n:
            # pad missing columns with zeros
            pad_count = model_n - cur_n
            for i in range(pad_count):
                features[f"_pad_{i}"] = 0.0
            warnings.append(f"Added {pad_count} zero-padding columns to match model expected features ({model_n}).")
        else:
            # trim extra columns
            features = features.iloc[:, :model_n]
            warnings.append(f"Trimmed input features from {cur_n} to {model_n} to match model expected features.")

    # Prefer probabilistic output when available
    try:
        preds = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(features)
                if probs.ndim == 2 and probs.shape[1] > 1:
                    preds = probs[:, 1]
                else:
                    preds = probs.ravel()
            except Exception:
                preds = None

        if preds is None:
            preds = model.predict(features)
    except Exception as exc:
        return {"error": f"Falha ao gerar previsões: {exc}", "warnings": warnings}

    result_df = df.copy()
    result_df["fadiga_analisada"] = preds

    # load threshold from summary if available
    threshold = None
    summ_path = locate.get("summary_path")
    model_summary = None
    if summ_path:
        try:
            with open(summ_path, "r", encoding="utf8") as fh:
                model_summary = json.load(fh)
                threshold = model_summary.get("alert_threshold")
        except Exception:
            model_summary = None

    if threshold is None:
        try:
            threshold = float(os.environ.get("ALERT_THRESHOLD", 0.5))
        except Exception:
            threshold = 0.5

    # simple summary
    peak_risk = float(result_df["fadiga_analisada"].max())
    peak_row = result_df.loc[result_df["fadiga_analisada"].idxmax()] if not result_df.empty else None
    peak_ts = None
    if peak_row is not None and "context_timestamp" in peak_row:
        peak_ts = str(peak_row.get("context_timestamp"))

    first_alert_ts = None
    alerted = result_df[result_df["fadiga_analisada"] >= threshold]
    if not alerted.empty and "context_timestamp" in alerted.columns:
        first_alert_ts = str(pd.to_datetime(alerted["context_timestamp"]).min())

    # top correlated features with risk (abs corr)
    numeric = features.select_dtypes(include=["number"]).copy()
    top_features = {}
    if not numeric.empty:
        try:
            corr = numeric.corrwith(result_df["fadiga_analisada"]).abs().sort_values(ascending=False)
            top_features = corr.head(10).dropna().to_dict()
        except Exception:
            top_features = {}

    # prepare result folder
    if result_root:
        result_root_path = Path(result_root)
    else:
        env_root = os.environ.get(RESULT_ROOT_ENV)
        if env_root:
            result_root_path = Path(env_root)
        else:
            # default to user's home data/result_campaigns for portability
            result_root_path = Path.home() / "data" / "result_campaigns"

    result_root_path.mkdir(parents=True, exist_ok=True)

    # Use fixed filenames per campaign so repeated runs overwrite the same artifact.
    parquet_path = result_root_path / f"{campaign_id}_analysis.parquet"
    csv_path = result_root_path / f"{campaign_id}_analysis.csv"
    summary_path = result_root_path / f"{campaign_id}_analysis_summary.json"

    try:
        # Try to save as parquet first (space efficient). If that fails, fallback to CSV.
        try:
            result_df.to_parquet(parquet_path, index=False)
            saved_parquet = True
        except Exception as pq_exc:
            saved_parquet = False

        # Always write a JSON summary referencing the saved artifact
        summary = {
            "campaign_id": campaign_id,
            "model_path": model_path,
            "model_dir": locate.get("model_dir"),
            "threshold": threshold,
            "peak_risk": peak_risk,
            "peak_ts": peak_ts,
            "first_alert_ts": first_alert_ts,
            "rows": int(len(result_df)),
            "top_features": top_features,
            "saved_at": datetime.utcnow().isoformat() + "Z",
        }
        # include model_summary if available
        if model_summary:
            summary["model_summary"] = model_summary

        if saved_parquet:
            summary["parquet"] = str(parquet_path)
        else:
            # fallback: write CSV
            result_df.to_csv(csv_path, index=False)
            summary["csv"] = str(csv_path)
            # record parquet error
            summary["parquet_error"] = str(pq_exc)

        with open(summary_path, "w", encoding="utf8") as fh:
            json.dump(summary, fh, indent=2, default=str)
    except Exception as exc:
        return {"error": f"Falha ao salvar resultados: {exc}"}

    out = {"summary": str(summary_path), "model_dir": locate.get("model_dir")}
    if saved_parquet:
        out["parquet"] = str(parquet_path)
    else:
        out["csv"] = str(csv_path)
    return out
