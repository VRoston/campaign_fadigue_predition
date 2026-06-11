"""Streamlit app for campaign prognosis prediction (72h)."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / ".env", override=False)

import streamlit as st
import os

# Page configuration
st.set_page_config(
    page_title="Prognóstico de Campanha 72h",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme
st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 32px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.header("📊 Navegação")
    page = st.radio(
        "Selecione uma página:",
        ["Resultados do Modelo", "Análise de Campanha"],
        index=0,
    )

# Import page renderers
if page == "Resultados do Modelo":
    from app.pages.model_results import render_model_results
    render_model_results()
elif page == "Análise de Campanha":
    from app.pages.campaign_analysis import render_campaign_analysis
    render_campaign_analysis()
