"""Model Results page - displays model metrics and feature importance."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from app.config import (
    HOLDOUT_METRICS,
    CV_METRICS,
    FEATURE_IMPORTANCE,
)


st.markdown("# 📈 Resultados do Modelo")
st.markdown("---")

# Metrics cards at the top
st.markdown("### Desempenho no Holdout Temporal")

metric_col1, metric_col2, metric_col3 = st.columns(3, gap="medium")

with metric_col1:
    st.metric(
        "🎯 ROC-AUC",
        f"{HOLDOUT_METRICS['roc_auc']:.4f}",
        help="Área sob a curva ROC no teste temporal (mais próximo de 1 é melhor)"
    )

with metric_col2:
    st.metric(
        "✓ Precision",
        f"{HOLDOUT_METRICS['precision']*100:.1f}%",
        help="De cada 10 alertas positivos, quantos estão corretos"
    )

with metric_col3:
    st.metric(
        "🎪 Recall",
        f"{HOLDOUT_METRICS['recall']*100:.1f}%",
        help="De cada 10 campanhas boas, quantas o modelo detecta"
    )

st.markdown("---")

# Feature importance and CV metrics
col_features, col_cv = st.columns(2, gap="large")

with col_features:
    st.markdown("### 🌟 Top 15 Features por Importância")

    top_features = FEATURE_IMPORTANCE[:15]
    feature_names = [f["feature"] for f in top_features]
    importances = [f["importance"] for f in top_features]

    fig = go.Figure(data=[
        go.Bar(
            y=feature_names,
            x=importances,
            orientation='h',
            marker=dict(
                color=importances,
                colorscale='Greens',
                showscale=False
            ),
            text=[f"{imp:.0f}" for imp in importances],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="O que o modelo considera mais importante",
        xaxis_title="Importância (gain)",
        yaxis_title="",
        height=500,
        margin=dict(l=200),
        showlegend=False,
    )
    fig.update_xaxes(showticklabels=False)

    st.plotly_chart(fig, use_container_width=True)

with col_cv:
    st.markdown("### 📊 Validação Cruzada (5 Folds)")

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

    gap = CV_METRICS["roc_auc_mean"] - HOLDOUT_METRICS["roc_auc"]
    if abs(gap) <= 0.05:
        st.info(
            f"**Gap CV → Holdout:** {gap:.4f}\n\n"
            f"Generalização saudável (gap ≤ 0.05) ✓"
        )
    else:
        st.warning(
            f"**Gap CV → Holdout:** {gap:.4f}\n\n"
            f"⚠️ Gap acima do esperado (> 0.05) — possível overfitting"
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(
            "AUC CV (média)",
            f"{CV_METRICS['roc_auc_mean']:.4f}",
            f"±{CV_METRICS['roc_auc_std']:.4f}"
        )
    with col_b:
        st.metric(
            "PR-AUC CV (média)",
            f"{CV_METRICS['pr_auc_mean']:.4f}",
            f"±{CV_METRICS['pr_auc_std']:.4f}"
        )

st.markdown("---")

# Confusion matrix
st.markdown("### 🔲 Matriz de Confusão (Holdout)")

tp = HOLDOUT_METRICS["tp"]
fp = HOLDOUT_METRICS["fp"]
fn = HOLDOUT_METRICS["fn"]
tn = HOLDOUT_METRICS["tn"]

cm_col1, cm_col2 = st.columns(2)

with cm_col1:
    st.markdown("#### Resultados Positivos")
    st.success(f"✓ **Acerto Positivo (TP):** {tp} campanhas")
    st.warning(f"⚠️ **Falso Alarme (FP):** {fp} campanhas")

with cm_col2:
    st.markdown("#### Resultados Negativos")
    st.success(f"✓ **Acerto Negativo (TN):** {tn} campanhas")
    st.error(f"✗ **Não Detectado (FN):** {fn} campanhas")

confusion_data = [
    [tn, fp],
    [fn, tp]
]

fig_cm = go.Figure(data=go.Heatmap(
    z=confusion_data,
    x=["Predito: Não", "Predito: Sim"],
    y=["Real: Não", "Real: Sim"],
    text=[[tn, fp], [fn, tp]],
    texttemplate="%{text}",
    colorscale="Blues",
    showscale=False
))

fig_cm.update_layout(
    title="Matriz de Confusão",
    xaxis_title="Predição do Modelo",
    yaxis_title="Realidade",
    height=350,
)

st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")

# Explanation section
st.markdown("### 📝 Sobre o Modelo")

tabs = st.tabs(["Descrição", "Metodologia", "Dataset"])

with tabs[0]:
    st.markdown(f"""
    #### O que o modelo faz?

    O modelo analisa o comportamento das **primeiras 72 horas** de uma campanha
    de Meta Ads e prevê se ela tem potencial de **atingir o target de eficiência** (CPA alvo).

    **Resultado:**
    - 🟢 **Campanha com Potencial**: Indicadores sugerem sucesso
    - 🔴 **Campanha em Risco**: Indicadores alertam para dificuldades

    **Precisão:**
    - De cada 10 alertas positivos, **{HOLDOUT_METRICS['precision']*10:.1f} estão corretos**
    - De cada 10 campanhas boas, **{HOLDOUT_METRICS['recall']*10:.1f} são detectadas**
    """)

with tabs[1]:
    st.markdown("""
    #### Metodologia

    **Tipo:** Classificador Binário LightGBM

    **Features:** 26 métricas calculadas nos primeiros 72h
    - Volume e spend
    - Eficiência (CPC, CPA, taxa de conversão)
    - Utilização de budget
    - CPA/CPC vs target configurado
    - Trajetória (tendências)
    - Configuração (objetivo, tipo de projeto)

    **Validação:**
    - Split Temporal: holdout = últimos 45 dias
    - GroupKFold CV: sem leakage por campanha
    - Threshold calibrado por F2-score (recall pesado)

    **Correções v2:**
    - Removidas features com viés de inferência (`age_max_hours`, `n_snaps_obs`)
    - Teste de ablação confirmou: sem impacto no AUC
    """)

with tabs[2]:
    ds = HOLDOUT_METRICS
    st.markdown("""
    #### Dataset de Treinamento

    **Período:** Novembro 2025 - Maio 2026

    **Campanhas:**
    - Desenvolvimento: 4.543 campanhas (44.6% sucesso)
    - Holdout: 1.266 campanhas (38.2% sucesso)

    **Fonte:** Plataforma OneClick Ads
    - Dados reais de campanhas Meta Ads
    - Snapshots horários
    - Features calculadas sem vazamento futuro
    """)

    ds_col1, ds_col2, ds_col3 = st.columns(3)
    with ds_col1:
        st.metric("Campanhas (holdout)", f"{ds['n_campanhas']:,}")
    with ds_col2:
        st.metric("Taxa de Sucesso (holdout)", f"{ds['pos_rate']*100:.1f}%")
    with ds_col3:
        st.metric("Período", "7 meses")
