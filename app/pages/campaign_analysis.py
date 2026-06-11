"""Campaign Analysis page - analyzes individual campaigns."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import glob
from pathlib import Path
from app.config import (
    DATA_PATH,
    MODEL,
    THRESHOLD,
    FEATURE_COLS,
    FEATURE_CATEGORIES,
    FEATURE_DISPLAY_NAMES,
)
from app.features import compute_features_72h, prepare_features_for_model


@st.cache_data(ttl=3600)
def load_campaign_ids():
    """Load unique campaign IDs from all parquet files."""
    campaign_ids = set()
    parquet_files = sorted(glob.glob(str(DATA_PATH / "*.parquet")))

    for f in parquet_files:
        try:
            df = pd.read_parquet(f, columns=["campaign_id"])
            campaign_ids.update(df["campaign_id"].unique())
        except Exception:
            pass

    return sorted(list(campaign_ids))


@st.cache_data(ttl=3600)
def load_campaign_snapshots(campaign_id: str):
    """Load all snapshots for a specific campaign."""
    parquet_files = sorted(glob.glob(str(DATA_PATH / "*.parquet")))
    snapshots = []

    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            campaign_data = df[df["campaign_id"] == campaign_id]
            if len(campaign_data) > 0:
                snapshots.append(campaign_data)
        except Exception:
            pass

    if not snapshots:
        return None

    df_campaign = pd.concat(snapshots, ignore_index=True)
    df_campaign["context_timestamp"] = pd.to_datetime(
        df_campaign["context_timestamp"], utc=True, errors="coerce"
    )
    df_campaign["campaign_startTime"] = pd.to_datetime(
        df_campaign["campaign_startTime"], utc=True, errors="coerce"
    )

    # Infer start time if missing
    if df_campaign["campaign_startTime"].isna().all():
        df_campaign["campaign_startTime"] = df_campaign["context_timestamp"].min()

    # Calculate age in hours
    df_campaign["age_hours"] = (
        (df_campaign["context_timestamp"] - df_campaign["campaign_startTime"]).dt.total_seconds() / 3600
    )

    # Filter to first 72h
    df_72h = df_campaign[
        (df_campaign["age_hours"] >= 0)
        & (df_campaign["age_hours"] <= 72)
        & (df_campaign["campaign_status"] == "ACTIVE")
    ].copy()

    return df_72h.sort_values("context_timestamp")


def calculate_probability_over_time(df_campaign):
    """Calculate model probability at each timestamp."""
    if len(df_campaign) < 3:
        return None

    probabilities = []
    timestamps = []
    ages = []

    for idx in range(len(df_campaign)):
        df_slice = df_campaign.iloc[:idx+1]
        try:
            features = compute_features_72h(df_slice)
            if len(features) > 0:
                X = prepare_features_for_model(features.to_dict(), FEATURE_COLS)
                prob = MODEL.predict_proba(X)[0, 1]
                probabilities.append(prob)
                timestamps.append(df_slice["context_timestamp"].iloc[-1])
                ages.append(df_slice["age_hours"].iloc[-1])
        except Exception:
            pass

    if len(probabilities) < 3:
        return None

    return pd.DataFrame({
        "timestamp": timestamps,
        "age_hours": ages,
        "probability": probabilities,
    })


def render_campaign_analysis():
    """Render the Campaign Analysis page."""
    st.title("🎯 Análise de Campanha")

    # Campaign selection
    col1, col2 = st.columns([3, 1])

    with col1:
        campaign_ids = load_campaign_ids()
        selected_campaign = st.selectbox(
            "Selecione uma campanha",
            campaign_ids,
            key="campaign_select",
        )

    with col2:
        analyze_button = st.button("Analisar", use_container_width=True)

    if not analyze_button or not selected_campaign:
        st.info("👈 Selecione uma campanha e clique em 'Analisar'")
        return

    # Load campaign data
    df_campaign = load_campaign_snapshots(selected_campaign)

    if df_campaign is None or len(df_campaign) == 0:
        st.error(f"Campanha {selected_campaign} não encontrada nos dados")
        return

    # Check minimum data quality
    if len(df_campaign) < 3:
        st.warning(
            f"⚠️ **Dados insuficientes para análise confiável**\n\n"
            f"Snapshots disponíveis: {len(df_campaign)} (mínimo: 3)"
        )
        return

    # Calculate features and prediction
    try:
        features = compute_features_72h(df_campaign)
        X = prepare_features_for_model(features.to_dict(), FEATURE_COLS)
        probability = MODEL.predict_proba(X)[0, 1]
        prediction = probability >= THRESHOLD
    except Exception as e:
        st.error(f"Erro ao processar campanha: {e}")
        return

    # Result block
    if prediction:
        result_color = "#1e7e34"
        icon = "✓"
        result_text = "**Campanha com potencial**"
        subtext = "Indicadores sugerem que a campanha atingirá o target de eficiência"
    else:
        result_color = "#b91c1c"
        icon = "✗"
        result_text = "**Campanha em risco**"
        subtext = "Indicadores alertam para dificuldade em atingir o target"

    # Display result with styling
    st.markdown(f"""
    <div style="
        background-color: {result_color};
        padding: 20px;
        border-radius: 8px;
        color: white;
        margin-bottom: 20px;
    ">
        <div style="font-size: 24px; margin-bottom: 10px;">{icon} {result_text}</div>
        <div style="font-size: 14px; margin-bottom: 15px;">{subtext}</div>
        <div style="font-size: 32px; font-weight: bold;">{probability*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics on the right
    col1, col2, col3 = st.columns(3)

    with col1:
        gasto_72h = features.get("spend_72h", np.nan)
        if not np.isnan(gasto_72h):
            st.metric("Gasto 72h", f"R$ {gasto_72h:,.0f}")
        else:
            st.metric("Gasto 72h", "—")

    with col2:
        spend_vs_budget = features.get("spend_vs_budget", np.nan)
        if not np.isnan(spend_vs_budget):
            st.metric("Spend vs Budget", f"{spend_vs_budget*100:.0f}%")
        else:
            st.metric("Spend vs Budget", "—")

    with col3:
        cpc = features.get("cpc_72h", np.nan)
        if not np.isnan(cpc):
            st.metric("CPC médio", f"R$ {cpc:.2f}")
        else:
            st.metric("CPC médio", "—")

    st.divider()

    # Probability evolution over time
    st.subheader("Evolução do Risco nas Primeiras 72h")

    prob_over_time = calculate_probability_over_time(df_campaign)

    if prob_over_time is not None and len(prob_over_time) > 2:
        fig = go.Figure()

        # Add probability line
        fig.add_trace(go.Scatter(
            x=prob_over_time["age_hours"],
            y=prob_over_time["probability"],
            mode="lines",
            name="Probabilidade",
            line=dict(color="#3b82f6", width=3),
            fill="tonexty",
            fillcolor="rgba(59, 130, 246, 0.2)",
        ))

        # Add threshold line
        fig.add_hline(
            y=THRESHOLD,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="Threshold",
            annotation_position="right",
        )

        fig.update_layout(
            title="Evolução do risco nas primeiras 72h",
            xaxis_title="Age (horas)",
            yaxis_title="P(Potencial)",
            height=400,
            template="plotly_dark",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Dados insuficientes para gráfico de evolução")

    st.divider()

    # Feature details
    with st.expander("📊 Ver detalhes das métricas calculadas"):
        st.markdown("**Acompanhamento das 26 features usadas pelo modelo:**")

        for category, feature_list in FEATURE_CATEGORIES.items():
            st.subheader(category)

            cat_features = []
            for feat in feature_list:
                value = features.get(feat, np.nan)
                display_name = FEATURE_DISPLAY_NAMES.get(feat, feat)

                if np.isnan(value) if isinstance(value, float) else False:
                    cat_features.append({
                        "Métrica": display_name,
                        "Valor": "—",
                        "Status": "⚠️ Ausente",
                    })
                else:
                    cat_features.append({
                        "Métrica": display_name,
                        "Valor": f"{value:.3f}" if isinstance(value, float) else str(value),
                        "Status": "✓",
                    })

            st.dataframe(pd.DataFrame(cat_features), use_container_width=True, hide_index=True)
