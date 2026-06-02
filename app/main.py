from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `from app.models...` works
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.models.predictor import enrich_with_fatigue_prediction, get_model_info
from app.utils.data_engine import (
    get_all_ad_accounts,
    get_campaigns_by_account,
    load_raw_campaign_history,
)
from app.ui.dashboard import render_schema_page, render_overview

st.set_page_config(page_title="Dashboard de Fadiga de Campanhas", layout="wide")
st.title("Dashboard de Fadiga de Campanhas")
st.caption("Leitura 100% local de arquivos Parquet usando DuckDB")


@st.cache_data(show_spinner=False)
def cached_ad_accounts() -> pd.DataFrame:
    return get_all_ad_accounts()


@st.cache_data(show_spinner=False)
def cached_campaigns(account_id: str) -> pd.DataFrame:
    return get_campaigns_by_account(account_id)


@st.cache_data(show_spinner=True)
def cached_campaign_history(campaign_id: str) -> pd.DataFrame:
    return load_raw_campaign_history(campaign_id)


accounts_df = cached_ad_accounts()

if accounts_df.empty:
    st.warning(
        "Nenhum arquivo parquet encontrado em data/raw_campaigns ou schema inválido."
    )
    st.stop()

# Allow the user to switch between the Dashboard and the Schema explorer
page = st.sidebar.radio("Página", ["Dashboard", "Schema"], index=0)

if page == "Schema":
    render_schema_page()
    st.stop()

account_options = accounts_df["adAccount_id"].tolist()
account_labels = {
    row["adAccount_id"]: f"{row['adAccount_name']} ({row['adAccount_id']})"
    for _, row in accounts_df.iterrows()
}

with st.sidebar:
    st.header("Filtros")
    selected_account_id = st.selectbox(
        "Ad Account",
        options=account_options,
        format_func=lambda value: account_labels.get(value, value),
    )

campaigns_df = cached_campaigns(selected_account_id)
if campaigns_df.empty:
    st.info("Nenhuma campanha encontrada para a ad account selecionada.")
    st.stop()

campaign_options = campaigns_df["campaign_id"].tolist()
campaign_labels = {
    row["campaign_id"]: f"{row['campaign_name']} ({row['campaign_id']})"
    for _, row in campaigns_df.iterrows()
}

with st.sidebar:
    selected_campaign_id = st.selectbox(
        "Campanha",
        options=campaign_options,
        format_func=lambda value: campaign_labels.get(value, value),
    )

try:
    history_df = cached_campaign_history(selected_campaign_id)
except Exception as exc:
    st.error(f"Falha ao carregar histórico da campanha: {exc}")
    st.stop()

if history_df.empty:
    st.info("Sem histórico para a campanha selecionada.")
    st.stop()

if "context_timestamp" in history_df.columns:
    history_df["context_timestamp"] = pd.to_datetime(
        history_df["context_timestamp"], errors="coerce", utc=True
    )

enriched_df = enrich_with_fatigue_prediction(history_df)

st.subheader(campaign_labels.get(selected_campaign_id, "Campanha"))

# Render the richer dashboard overview using componentized UI
model_info = get_model_info()
render_overview(enriched_df, model_info=model_info, campaign_id=selected_campaign_id)

with st.expander("Visualizar dados brutos"):
    st.dataframe(enriched_df, use_container_width=True)
