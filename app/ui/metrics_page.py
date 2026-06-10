"""Model metrics page - displays model performance and feature importance."""

import json
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from app.config import get_model_summary_path


@st.cache_data(show_spinner=False)
def load_model_summary() -> Optional[dict]:
    """Load model_summary.json from disk."""
    summary_path = get_model_summary_path()
    if not summary_path or not summary_path.exists():
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def render_metrics_page() -> None:
    """Render the model metrics page."""
    st.header("Métricas do Modelo")

    summary = load_model_summary()
    if not summary:
        st.warning(
            "Arquivo model_summary.json não encontrado. "
            "Coloque o arquivo em `/models/v4_lagged/model_summary.json`."
        )
        return

    # --- KPI Cards ---
    st.subheader("Performance no Holdout")
    cols = st.columns(3)

    # ROC-AUC Holdout
    roc_auc = summary.get("metrics_holdout_auc", {}).get("roc_auc")
    if roc_auc is not None:
        cols[0].metric(
            "ROC-AUC Holdout",
            f"{float(roc_auc):.3f}",
            help="Área sob a curva ROC no conjunto de holdout",
        )
    else:
        cols[0].metric("ROC-AUC Holdout", "—")

    # Recall
    recall = summary.get("metrics_holdout_auc", {}).get("recall")
    if recall is not None:
        cols[1].metric(
            "Recall",
            f"{float(recall) * 100:.1f}%",
            help="Proporção de campanhas com fadiga corretamente identificadas",
        )
    else:
        cols[1].metric("Recall", "—")

    # Gap CV → Holdout
    cv_auc = summary.get("metrics_cv", {}).get("mean_auc")
    holdout_auc = summary.get("metrics_holdout_auc", {}).get("roc_auc")
    if cv_auc is not None and holdout_auc is not None:
        gap = float(cv_auc) - float(holdout_auc)
        cols[2].metric(
            "Gap CV → Holdout",
            f"{gap:.3f}",
            help="Diferença entre AUC médio CV e AUC holdout (menor é melhor)",
        )
    else:
        cols[2].metric("Gap CV → Holdout", "—")

    st.markdown("---")

    # --- Feature Importance ---
    st.subheader("Importância das Features (Top 10)")
    feature_importance = summary.get("feature_importance")
    if feature_importance:
        # Sort by importance (descending) and take top 10
        fi_list = list(feature_importance.items())
        fi_list.sort(key=lambda x: float(x[1]), reverse=True)
        top_10 = fi_list[:10]

        feature_names = [f[0] for f in top_10]
        importance_values = [float(f[1]) for f in top_10]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=importance_values,
                    y=feature_names,
                    orientation="h",
                    marker=dict(color="rgba(99, 110, 250, 0.8)"),
                )
            ]
        )
        fig.update_layout(
            title="Features com maior importância",
            xaxis_title="Importância (ganho)",
            yaxis_title="Feature",
            height=400,
            showlegend=False,
            margin=dict(l=200),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance não disponível no model_summary.json")

    st.markdown("---")

    # --- CV Folds Results ---
    st.subheader("Resultados dos 5 Folds (Validação Cruzada)")
    cv_folds = summary.get("cv_folds")
    if cv_folds:
        # Convert to DataFrame for display
        fold_data = []
        for fold_idx, fold_info in enumerate(cv_folds, 1):
            fold_data.append({
                "Fold": fold_idx,
                "AUC": f"{float(fold_info.get('auc', 0)):.3f}",
                "PR-AUC": f"{float(fold_info.get('pr_auc', 0)):.3f}",
            })

        df_folds = pd.DataFrame(fold_data)
        st.dataframe(df_folds, use_container_width=False, hide_index=True)

        # Summary row
        st.caption(
            f"Média: AUC={float(summary.get('metrics_cv', {}).get('mean_auc', 0)):.3f} "
            f"(±{float(summary.get('metrics_cv', {}).get('std_auc', 0)):.3f})"
        )
    else:
        st.info("Dados dos folds não disponíveis no model_summary.json")

    st.markdown("---")

    # --- Model Info ---
    st.subheader("Informações do Modelo")
    col1, col2 = st.columns(2)
    col1.metric("Versão", summary.get("version", "—"))
    col2.metric("Data de Treino", summary.get("last_trained", "—"))
