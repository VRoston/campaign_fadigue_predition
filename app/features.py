"""Feature engineering for 72h prognosis prediction model."""

import numpy as np
import pandas as pd


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
    if len(df) == 0:
        return pd.Series(dtype=float)

    grp = df.sort_values('age_hours').reset_index(drop=True)
    n = len(grp)

    # Convert to numeric
    spend3d = to_num(grp['campaign_insights_spend_last3d'])
    spend_today = to_num(grp.get('campaign_insights_spend_today', pd.Series(dtype=float)))
    budget = to_num(grp['campaign_dailyBudget'])

    cpc3d = to_num(grp['campaign_insights_cpcWeightedAverageAllAdSets_last3d'])
    cpa3d_m = to_num(grp['campaign_insights_ocpp_last3d'])
    cpa3d_u = to_num(grp['campaign_insights_ocppUtm_last3d'])
    cc3d = to_num(grp['campaign_insights_ccWeightedAverageAllAdSets_last3d'])
    nic3d = to_num(grp['campaign_insights_nicSumAllAdSets_last3d'])
    nc3d = to_num(grp['campaign_insights_ncSumAllAdSets_last3d'])

    cpabe = to_num(grp['input_cpabe']).median()
    tm_val = to_num(grp['input_tm']).median()
    target_eff = cpabe if not np.isnan(cpabe) else tm_val

    active_ads = to_num(grp['campaign_insights_activeAdsCount'])
    total_ads = to_num(grp['campaign_insights_adsCount'])
    active_adsets = to_num(grp['campaign_insights_activeAdSetsCount'])

    obj = str(grp['campaign_objective'].iloc[-1]).upper() if len(grp) > 0 else ""
    bopt = str(grp['campaign_budgetOptimization'].iloc[-1]).upper() if len(grp) > 0 else ""
    ptype = str(grp.get('adAccountTracking_config_projectType', pd.Series(dtype=str)).iloc[-1] if len(grp) > 0 else "").lower()

    # A) Volume and spend
    spend_max = spend3d.max()
    budget_med = budget.median()
    budget_3d = budget_med * 3
    spend_vs_budget = spend_max / budget_3d if budget_3d > 0 else np.nan

    spend_daily_avg = spend_max / 3.0
    burn_rate = spend_daily_avg / budget_med if budget_med > 0 else np.nan

    # B) Efficiency
    cpc_med = cpc3d.median()
    cpa_med = cpa3d_u.median() if cpa3d_u.notna().any() else cpa3d_m.median()
    cc_med = cc3d.median()
    nic_max = nic3d.max()
    nc_max = nc3d.max()

    nic_per_100 = (nic_max / spend_max * 100) if (spend_max and spend_max > 0) else np.nan

    # C) CPA/CPC vs target
    cpa_vs_target = cpa_med / target_eff if (target_eff and target_eff > 0 and not np.isnan(cpa_med)) else np.nan
    cpc_vs_target = cpc_med / target_eff if (target_eff and target_eff > 0 and not np.isnan(cpc_med)) else np.nan
    budget_vs_target = budget_med / target_eff if (target_eff and target_eff > 0) else np.nan

    # D) Trajectory (first 36h vs last 36h)
    mid = grp['age_hours'].max() / 2.0
    grp_early = grp[grp['age_hours'] <= mid]
    grp_late = grp[grp['age_hours'] > mid]

    cpc_early = to_num(grp_early['campaign_insights_cpcWeightedAverageAllAdSets_last3d']).median()
    cpc_late = to_num(grp_late['campaign_insights_cpcWeightedAverageAllAdSets_last3d']).median()
    cpc_trend = (cpc_late / cpc_early) if (not np.isnan(cpc_early) and not np.isnan(cpc_late) and cpc_early > 0) else np.nan

    spend_early = to_num(grp_early['campaign_insights_spend_last3d']).max()
    spend_late = to_num(grp_late['campaign_insights_spend_last3d']).max()
    spend_trend = (spend_late - spend_early) / (spend_early + 1e-6) if not np.isnan(spend_early) else np.nan

    # E) Configuration
    pct_ads_active = (active_ads.median() / total_ads.median()) if (not np.isnan(total_ads.median()) and total_ads.median() > 0) else np.nan

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
        'active_ads': active_ads.median(),
        'pct_ads_active': pct_ads_active,
        'active_adsets': active_adsets.median(),
        'is_cbo': int(bopt == 'CBO'),
        'is_sales': int(obj == 'OUTCOME_SALES'),
        'is_leads': int(obj == 'OUTCOME_LEADS'),
        'is_engagement': int(obj == 'OUTCOME_ENGAGEMENT'),
        'is_ecommerce': int(ptype == 'ecommerce'),
        'is_infoprodutos': int(ptype == 'infoprodutos'),
    }

    return pd.Series(features)


def prepare_features_for_model(features_dict: dict, feature_names: list) -> np.ndarray:
    """
    Select and order exactly the columns expected by the model.
    Missing features are filled with NaN.
    """
    features = []
    for name in feature_names:
        features.append(features_dict.get(name, np.nan))
    return np.array(features, dtype='float32').reshape(1, -1)
