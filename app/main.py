"""Main Streamlit app for campaign fatigue prediction."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from app.ui.metrics_page import render_metrics_page
from app.ui.campaign_page import render_campaign_page

# Page configuration
st.set_page_config(
    page_title="Predição de Fadiga - Meta Ads",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Predição de Fadiga de Campanhas")
st.caption("LightGBM v4 | Modelo de risco com 48h de antecedência")

# Sidebar navigation
with st.sidebar:
    st.header("Navegação")
    page = st.radio(
        "Selecione uma página:",
        ["Métricas do Modelo", "Análise de Campanha"],
        index=0,
    )

# Render selected page
if page == "Métricas do Modelo":
    render_metrics_page()
elif page == "Análise de Campanha":
    render_campaign_page()
