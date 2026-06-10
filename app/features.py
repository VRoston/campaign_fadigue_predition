"""Feature engineering for fatigue prediction model."""

import pandas as pd
import numpy as np


def compute_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lagged features (24h lag) from raw campaign snapshots.

    Expects columns like:
    - context_timestamp
    - campaign_insights_spend_today (or similar spend columns)
    - campaign_impressions (or similar impression columns)
    - campaign_clicks (or similar click columns)
    - campaign_conversions (or similar conversion columns)

    Computes:
    - spend_lag24h
    - impressions_lag24h
    - clicks_lag24h
    - conversions_lag24h
    - cpc (cost per click)
    - ctr (click-through rate)
    - cpa (cost per action/conversion)
    - budget_spent_ratio
    """
    features = df.copy()

    # Ensure timestamp is datetime
    if "context_timestamp" in features.columns:
        features["context_timestamp"] = pd.to_datetime(
            features["context_timestamp"], errors="coerce", utc=True
        )

    # Define candidate column names for each metric
    spend_candidates = [
        "spend", "campaign_insights_spend_today",
        "adAccount_insights_spend_today", "campaign_spend"
    ]
    impressions_candidates = [
        "impressions", "campaign_impressions",
        "campaign_insights_impressions"
    ]
    clicks_candidates = [
        "clicks", "campaign_clicks", "campaign_insights_clicks"
    ]
    conversions_candidates = [
        "conversions", "campaign_conversions",
        "campaign_insights_actions", "campaign_actions"
    ]

    def _find_column(candidates):
        """Find first available column from candidates."""
        for col in candidates:
            if col in features.columns:
                return col
        return None

    spend_col = _find_column(spend_candidates)
    impr_col = _find_column(impressions_candidates)
    clicks_col = _find_column(clicks_candidates)
    conv_col = _find_column(conversions_candidates)

    # Compute 24h lags
    if spend_col:
        features["spend_lag24h"] = (
            features[spend_col]
            .astype(float, errors="coerce")
            .fillna(0)
            .rolling(window="24h", on="context_timestamp", closed="left")
            .sum()
        )

    if impr_col:
        features["impressions_lag24h"] = (
            features[impr_col]
            .astype(float, errors="coerce")
            .fillna(0)
            .rolling(window="24h", on="context_timestamp", closed="left")
            .sum()
        )

    if clicks_col:
        features["clicks_lag24h"] = (
            features[clicks_col]
            .astype(float, errors="coerce")
            .fillna(0)
            .rolling(window="24h", on="context_timestamp", closed="left")
            .sum()
        )

    if conv_col:
        features["conversions_lag24h"] = (
            features[conv_col]
            .astype(float, errors="coerce")
            .fillna(0)
            .rolling(window="24h", on="context_timestamp", closed="left")
            .sum()
        )

    # Compute ratios (CPC, CTR, CPA)
    if spend_col and clicks_col:
        clicks_numeric = pd.to_numeric(features[clicks_col], errors="coerce").fillna(0)
        spend_numeric = pd.to_numeric(features[spend_col], errors="coerce").fillna(0)
        # CPC: Cost Per Click
        features["cpc"] = np.where(
            clicks_numeric > 0,
            spend_numeric / clicks_numeric,
            0
        )

    if impr_col and clicks_col:
        impr_numeric = pd.to_numeric(features[impr_col], errors="coerce").fillna(0)
        clicks_numeric = pd.to_numeric(features[clicks_col], errors="coerce").fillna(0)
        # CTR: Click-Through Rate
        features["ctr"] = np.where(
            impr_numeric > 0,
            clicks_numeric / impr_numeric,
            0
        )

    if spend_col and conv_col:
        conv_numeric = pd.to_numeric(features[conv_col], errors="coerce").fillna(0)
        spend_numeric = pd.to_numeric(features[spend_col], errors="coerce").fillna(0)
        # CPA: Cost Per Action
        features["cpa"] = np.where(
            conv_numeric > 0,
            spend_numeric / conv_numeric,
            0
        )

    # Budget pressure ratio
    daily_budget_candidates = [
        "daily_budget", "campaign_dailyBudget", "campaign_daily_budget"
    ]
    daily_budget_col = _find_column(daily_budget_candidates)
    if spend_col and daily_budget_col:
        spend_numeric = pd.to_numeric(features[spend_col], errors="coerce").fillna(0)
        budget_numeric = pd.to_numeric(features[daily_budget_col], errors="coerce").fillna(1)
        features["budget_spent_ratio"] = np.where(
            budget_numeric > 0,
            spend_numeric / budget_numeric,
            0
        )

    # Fill NaN with 0
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].fillna(0)

    return features


def prepare_features_for_model(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Prepare features for model prediction by selecting and ordering columns.

    Args:
        df: DataFrame with computed features
        feature_names: List of feature names expected by the model

    Returns:
        DataFrame with exactly the required features in the correct order
    """
    features = df[feature_names].copy()
    # Ensure numeric and fill NaN
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0)
    return features
