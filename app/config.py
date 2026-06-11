"""Configuration for the prognosis prediction app."""

import os
import json
from pathlib import Path
import joblib

# Paths from environment
DATA_PATH = Path(os.getenv("CAMPAIGN_DATA_PATH", "/home/vroston/data/raw_campaigns"))
MODEL_ROOT = Path(os.getenv("CAMPAIGN_MODEL_ROOT", "/home/vroston/data/models"))
MODEL_DIR = MODEL_ROOT / "prognosis_72h"

# Model and data files
MODEL_FILE = MODEL_DIR / "lgbm_prognosis_72h.joblib"
SUMMARY_FILE = MODEL_DIR / "model_summary.json"
DATASET_FILE = MODEL_DIR / "dataset_72h.parquet"

# Load model summary
with open(SUMMARY_FILE) as f:
    MODEL_SUMMARY = json.load(f)

# Load model
MODEL = joblib.load(MODEL_FILE)

# Extract key info from summary
THRESHOLD = MODEL_SUMMARY["threshold"]
FEATURE_COLS = MODEL_SUMMARY["feature_cols"]
HOLDOUT_METRICS = MODEL_SUMMARY["holdout"]
CV_METRICS = MODEL_SUMMARY["cv"]
FEATURE_IMPORTANCE = MODEL_SUMMARY["feature_importance"]

# Feature categories
FEATURE_CATEGORIES = {
    "Volume": [
        "spend_72h",
        "spend_daily_avg",
        "nc_72h",
        "nic_72h",
    ],
    "Eficiência": [
        "cpc_72h",
        "cpa_72h",
        "cc_72h",
        "nic_per_100_spend",
    ],
    "Utilização": [
        "budget_daily",
        "spend_vs_budget",
        "burn_rate",
    ],
    "Target": [
        "budget_vs_target",
        "cpa_vs_target",
        "cpc_vs_target",
        "target_configured",
    ],
    "Trajetória": [
        "cpc_trend_72h",
        "spend_trend_72h",
    ],
    "Configuração": [
        "active_ads",
        "pct_ads_active",
        "active_adsets",
        "is_cbo",
        "is_sales",
        "is_leads",
        "is_engagement",
        "is_ecommerce",
        "is_infoprodutos",
    ],
}

# Display names for features
FEATURE_DISPLAY_NAMES = {
    "spend_72h": "Gasto 72h (R$)",
    "spend_daily_avg": "Gasto diário médio (R$)",
    "nc_72h": "Cliques totais",
    "nic_72h": "Conversões totais",
    "cpc_72h": "CPC médio (R$)",
    "cpa_72h": "CPA médio (R$)",
    "cc_72h": "Taxa de conversão (%)",
    "nic_per_100_spend": "Conv. por R$ 100",
    "budget_daily": "Budget diário (R$)",
    "spend_vs_budget": "Gasto vs Budget (%)",
    "burn_rate": "Taxa de queima",
    "budget_vs_target": "Budget vs Target",
    "cpa_vs_target": "CPA vs Target",
    "cpc_vs_target": "CPC vs Target",
    "target_configured": "Target configurado",
    "cpc_trend_72h": "Tendência CPC",
    "spend_trend_72h": "Tendência Spend",
    "active_ads": "Anúncios ativos",
    "pct_ads_active": "% Anúncios ativos",
    "active_adsets": "Adsets ativos",
    "is_cbo": "CBO ativo",
    "is_sales": "Objetivo: Vendas",
    "is_leads": "Objetivo: Leads",
    "is_engagement": "Objetivo: Engajamento",
    "is_ecommerce": "Tipo: E-commerce",
    "is_infoprodutos": "Tipo: Infoprodutos",
}
