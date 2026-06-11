"""Feature engineering for 72h prognosis prediction model."""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def to_num(s):
    """Convert to numeric, coercing errors to NaN."""
    return pd.to_numeric(s, errors='coerce')


def safe_ratio(num, den, cap=10.0):
    """Safe division with clipping."""
    mask = (den.notna()) & (den > 1e-6) & (num.notna())
    result = np.where(mask, np.clip(num / den, 0, cap), np.nan)
    return pd.Series(result, index=num.index)


def compute_features_72h(df: pd.DataFrame) -> pd.Series:
    """
    Compute all 26 features from first 72h snapshots.

    Input: DataFrame with campaign snapshots, sorted by age_hours (0-72).
    Output: Series with all 26 feature values (indexed by feature name).
    """
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    try:
        grp = df.copy()
        if "age_hours" in grp.columns:
            grp = grp.sort_values("age_hours").reset_index(drop=True)
        else:
            grp = grp.reset_index(drop=True)
    except Exception:
        logger.exception("compute_features_72h: failed to sort/reset index")
        return pd.Series(dtype=float)

    # Convert to numeric - handle missing columns safely
    spend3d = to_num(grp.get('campaign_insights_spend_last3d', pd.Series(dtype=float)))
    budget = to_num(grp.get('campaign_dailyBudget', pd.Series(dtype=float)))

    cpc3d = to_num(grp.get('campaign_insights_cpcWeightedAverageAllAdSets_last3d', pd.Series(dtype=float)))
    cpa3d_m = to_num(grp.get('campaign_insights_ocpp_last3d', pd.Series(dtype=float)))
    cpa3d_u = to_num(grp.get('campaign_insights_ocppUtm_last3d', pd.Series(dtype=float)))
    cc3d = to_num(grp.get('campaign_insights_ccWeightedAverageAllAdSets_last3d', pd.Series(dtype=float)))
    nic3d = to_num(grp.get('campaign_insights_nicSumAllAdSets_last3d', pd.Series(dtype=float)))
    nc3d = to_num(grp.get('campaign_insights_ncSumAllAdSets_last3d', pd.Series(dtype=float)))

    cpabe = to_num(grp.get('input_cpabe', pd.Series(dtype=float))).median()
    tm_val = to_num(grp.get('input_tm', pd.Series(dtype=float))).median()
    target_eff = cpabe if not np.isnan(cpabe) else tm_val

    active_ads = to_num(grp.get('campaign_insights_activeAdsCount', pd.Series(dtype=float)))
    total_ads = to_num(grp.get('campaign_insights_adsCount', pd.Series(dtype=float)))
    active_adsets = to_num(grp.get('campaign_insights_activeAdSetsCount', pd.Series(dtype=float)))

    # Safe column access for string columns — avoid IndexError on empty Series
    obj_col = grp.get('campaign_objective')
    obj = str(obj_col.iloc[-1]).upper() if (obj_col is not None and len(obj_col) > 0) else ""

    bopt_col = grp.get('campaign_budgetOptimization')
    bopt = str(bopt_col.iloc[-1]).upper() if (bopt_col is not None and len(bopt_col) > 0) else ""

    ptype_col = grp.get('adAccountTracking_config_projectType')
    if ptype_col is not None and len(ptype_col) > 0 and ptype_col.iloc[-1] is not None:
        ptype = str(ptype_col.iloc[-1]).lower()
    else:
        ptype = ""

    try:
        # A) Volume and spend
        spend_max = spend3d.max() if spend3d.notna().any() else np.nan
        budget_med = budget.median() if budget.notna().any() else np.nan
        budget_3d = budget_med * 3 if not np.isnan(budget_med) else 1
        spend_vs_budget = spend_max / budget_3d if (not np.isnan(spend_max) and budget_3d > 0) else np.nan

        spend_daily_avg = spend_max / 3.0 if not np.isnan(spend_max) else np.nan
        burn_rate = (
            spend_daily_avg / budget_med
            if (not np.isnan(spend_daily_avg) and not np.isnan(budget_med) and budget_med > 0)
            else np.nan
        )

        # B) Efficiency
        cpc_med = cpc3d.median() if cpc3d.notna().any() else np.nan
        cpa_med = cpa3d_u.median() if cpa3d_u.notna().any() else cpa3d_m.median()
        cc_med = cc3d.median() if cc3d.notna().any() else np.nan
        nic_max = nic3d.max() if nic3d.notna().any() else np.nan
        nc_max = nc3d.max() if nc3d.notna().any() else np.nan

        nic_per_100 = (
            (nic_max / spend_max * 100)
            if (not np.isnan(nic_max) and not np.isnan(spend_max) and spend_max > 0)
            else np.nan
        )

        # C) CPA/CPC vs target
        cpa_vs_target = (
            cpa_med / target_eff
            if (not np.isnan(target_eff) and target_eff > 0 and not np.isnan(cpa_med))
            else np.nan
        )
        cpc_vs_target = (
            cpc_med / target_eff
            if (not np.isnan(target_eff) and target_eff > 0 and not np.isnan(cpc_med))
            else np.nan
        )
        budget_vs_target = (
            budget_med / target_eff
            if (not np.isnan(target_eff) and target_eff > 0 and not np.isnan(budget_med))
            else np.nan
        )

        # D) Trajectory (first half vs second half of the 72h window)
        if "age_hours" in grp.columns and len(grp) > 1:
            mid = grp['age_hours'].max() / 2.0
            grp_early = grp[grp['age_hours'] <= mid]
            grp_late = grp[grp['age_hours'] > mid]

            cpc_early = to_num(
                grp_early.get('campaign_insights_cpcWeightedAverageAllAdSets_last3d', pd.Series(dtype=float))
            ).median()
            cpc_late = to_num(
                grp_late.get('campaign_insights_cpcWeightedAverageAllAdSets_last3d', pd.Series(dtype=float))
            ).median()
            cpc_trend = (
                (cpc_late / cpc_early)
                if (not np.isnan(cpc_early) and not np.isnan(cpc_late) and cpc_early > 0)
                else np.nan
            )

            spend_early = to_num(
                grp_early.get('campaign_insights_spend_last3d', pd.Series(dtype=float))
            ).max()
            spend_late = to_num(
                grp_late.get('campaign_insights_spend_last3d', pd.Series(dtype=float))
            ).max()
            spend_trend = (
                (spend_late - spend_early) / (spend_early + 1e-6)
                if (not np.isnan(spend_early) and not np.isnan(spend_late) and spend_early > 0)
                else np.nan
            )
        else:
            cpc_trend = np.nan
            spend_trend = np.nan

        # E) Configuration
        active_ads_med = active_ads.median() if active_ads.notna().any() else np.nan
        total_ads_med = total_ads.median() if total_ads.notna().any() else np.nan
        pct_ads_active = (
            (active_ads_med / total_ads_med)
            if (not np.isnan(active_ads_med) and not np.isnan(total_ads_med) and total_ads_med > 0)
            else np.nan
        )
        active_adsets_med = active_adsets.median() if active_adsets.notna().any() else np.nan

        features = {
            'spend_72h': spend_max,
            'spend_daily_avg': spend_daily_avg,
            'nc_72h': nc_max,
            'nic_72h': nic_max,
            'cpc_72h': cpc_med,
            'cpa_72h': cpa_med,
            'cc_72h': cc_med,
            'nic_per_100_spend': nic_per_100,
            'budget_daily': budget_med,
            'spend_vs_budget': spend_vs_budget,
            'burn_rate': burn_rate,
            'budget_vs_target': budget_vs_target,
            'cpa_vs_target': cpa_vs_target,
            'cpc_vs_target': cpc_vs_target,
            'target_configured': int(not np.isnan(target_eff)),
            'cpc_trend_72h': cpc_trend,
            'spend_trend_72h': spend_trend,
            'active_ads': active_ads_med,
            'pct_ads_active': pct_ads_active,
            'active_adsets': active_adsets_med,
            'is_cbo': int(bopt == 'CBO'),
            'is_sales': int(obj == 'OUTCOME_SALES'),
            'is_leads': int(obj == 'OUTCOME_LEADS'),
            'is_engagement': int(obj == 'OUTCOME_ENGAGEMENT'),
            'is_ecommerce': int(ptype == 'ecommerce'),
            'is_infoprodutos': int(ptype == 'infoprodutos'),
        }

        return pd.Series(features)

    except Exception:
        logger.exception("compute_features_72h: unexpected error during feature computation")
        return pd.Series(dtype=float)


def prepare_features_for_model(features_dict: dict, feature_names: list) -> np.ndarray:
    """
    Select and order exactly the columns expected by the model.
    Missing features are filled with NaN.
    """
    features = [features_dict.get(name, np.nan) for name in feature_names]
    return np.array(features, dtype='float32').reshape(1, -1)
