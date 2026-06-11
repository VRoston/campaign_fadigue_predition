"""Main Streamlit app for campaign prognosis prediction."""

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

# Page configuration - must be called first
st.set_page_config(
    page_title="Prognóstico de Campanha 72h",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None,
)

from app.config import MODEL_SUMMARY, HOLDOUT_METRICS

_ds = MODEL_SUMMARY.get("dataset", {})
_n_total = _ds.get("n_total", "—")
_auc = HOLDOUT_METRICS.get("roc_auc", 0)
_precision = HOLDOUT_METRICS.get("precision", 0) * 100
_recall = HOLDOUT_METRICS.get("recall", 0) * 100

# Main page
st.markdown("# 🚀 Prognóstico de Campanha 72h")
st.markdown("---")

st.markdown(f"""
## Bem-vindo!

Este dashboard analisa campanhas de Meta Ads e prevê se elas têm potencial
de atingir o target de eficiência nos **primeiros 72 horas** de operação.

### 📊 O Modelo
- **Tipo:** Classificador Binário LightGBM
- **Features:** 26 métricas das primeiras 72h
- **Desempenho:** ROC-AUC {_auc:.4f}, Precision {_precision:.1f}%, Recall {_recall:.1f}%
- **Dataset:** {_n_total:,} campanhas reais (nov/2025 - mai/2026)

### 🎯 Como Usar
1. **Resultados do Modelo** → Veja as métricas, feature importance e validação
2. **Análise de Campanha** → Analise uma campanha específica

### 📈 Fluxo de Análise
- Selecione uma Ad Account
- Escolha uma campanha
- Clique em "Analisar"
- Veja a probabilidade e métricas detalhadas

---

**Navegue usando o menu ao lado →**
""")

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px; margin-top: 50px;">
Desenvolvido com Streamlit + LightGBM
</div>
""", unsafe_allow_html=True)
