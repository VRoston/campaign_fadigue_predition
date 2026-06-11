"""Model Results page - displays model metrics and feature importance."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from app.config import (
    HOLDOUT_METRICS,
    CV_METRICS,
    FEATURE_IMPORTANCE,
)


def render_model_results():
    """Render the Model Results page."""
    st.title("📈 Resultados do Modelo")

    # Metrics cards at the top
    st.subheader("Desempenho no Holdout")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "ROC-AUC",
            f"{HOLDOUT_METRICS['roc_auc']:.4f}",
            help="Área sob a curva ROC no teste temporal"
        )

    with col2:
        st.metric(
            "Precision",
            f"{HOLDOUT_METRICS['precision']*100:.1f}%",
            help="Acurácia dos alertas positivos"
        )

    with col3:
        st.metric(
            "Recall",
            f"{HOLDOUT_METRICS['recall']*100:.1f}%",
            help="Campanhas boas detectadas"
        )

    st.divider()

    # Feature importance and CV metrics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 15 Features por Importância")

        # Get top 15 features
        top_features = FEATURE_IMPORTANCE[:15]
        feature_names = [f["feature"] for f in top_features]
        importances = [f["importance"] for f in top_features]

        # Create horizontal bar chart
        fig = go.Figure(data=[
            go.Bar(
                y=feature_names,
                x=importances,
                orientation='h',
                marker=dict(color='#22c55e'),
                text=[f"{imp:.0f}" for imp in importances],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="O que o modelo considera",
            xaxis_title="Importância (gain)",
            yaxis_title="",
            height=500,
            margin=dict(l=200),
            showlegend=False,
            template="plotly_dark",
        )
        fig.update_xaxes(showticklabels=False)

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Cross-Validation (5 Folds)")

        # Create CV results table
        cv_folds = CV_METRICS["folds"]
        cv_df = pd.DataFrame([
            {
                "Fold": f["fold"],
                "AUC": f"{f['roc_auc']:.4f}",
                "PR-AUC": f"{f['pr_auc']:.4f}",
                "Iterações": int(f["best_iter"]),
            }
            for f in cv_folds
        ])

        st.dataframe(cv_df, use_container_width=True, hide_index=True)

        # Gap information
        gap = CV_METRICS["roc_auc_mean"] - HOLDOUT_METRICS["roc_auc"]
        st.caption(f"Gap CV → Holdout: {gap:.4f} — generalização saudável")

    st.divider()

    # Confusion matrix
    st.subheader("Matriz de Confusão (Holdout)")

    tp = HOLDOUT_METRICS["tp"]
    fp = HOLDOUT_METRICS["fp"]
    fn = HOLDOUT_METRICS["fn"]
    tn = HOLDOUT_METRICS["tn"]

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"✓ **Acerto Positivo**: {tp}")
        st.success(f"✓ **Acerto Negativo**: {tn}")

    with col2:
        st.warning(f"⚠ **Falso Alarme**: {fp}")
        st.error(f"✗ **Não Detectado**: {fn}")

    st.divider()

    # Explanation
    st.markdown("""
    ### 📝 O que o modelo faz

    O modelo analisa o comportamento das **primeiras 72 horas** de uma campanha
    e prevê se ela tem potencial de **atingir o target de eficiência**
    (CPA alvo configurado).

    **Dataset de treinamento:**
    - Período: nov/2025 a abr/2026
    - Campanhas: 4.543 no desenvolvimento, 1.266 no teste
    - Taxa de sucesso: 38% das campanhas chegam ao target

    **Metodologia:**
    - Classificador LightGBM com 26 features
    - Split temporal: holdout são campanhas dos últimos 45 dias
    - GroupKFold CV: sem leakage por campanha
    - Threshold calibrado por F2-score (recall pesado)
    """)
