"""Campaign analysis page - predict fatigue and analyze trends."""

import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.config import get_data_path, get_model_path, get_alert_threshold
from app.features import compute_lagged_features, prepare_features_for_model
from app.utils.data_engine import load_raw_campaign_history


@st.cache_data(show_spinner=False)
def load_model():
    """Load the LightGBM fatigue model."""
    model_path = get_model_path()
    if not model_path or not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_campaign_ids():
    """Get list of campaign IDs from parquet files."""
    data_path = get_data_path()
    parquet_files = list(data_path.glob("*.parquet"))

    if not parquet_files:
        return []

    try:
        # Read all parquets and extract unique campaign IDs
        dfs = []
        for pf in parquet_files[:10]:  # Limit to avoid memory issues
            try:
                df = pd.read_parquet(pf)
                if "campaign_id" in df.columns:
                    dfs.append(df[["campaign_id", "campaign_name"]].drop_duplicates())
            except Exception:
                continue

        if dfs:
            combined = pd.concat(dfs, ignore_index=True).drop_duplicates()
            return combined.sort_values("campaign_name").to_dict("records")
        return []
    except Exception:
        return []


def render_campaign_page() -> None:
    """Render the campaign analysis page."""
    st.header("Análise de Campanha")

    # Get list of campaigns
    campaigns = get_campaign_ids()
    if not campaigns:
        st.warning(
            "Nenhuma campanha encontrada. "
            "Verifique se os parquets estão em `/data/raw`."
        )
        return

    # Dropdown to select campaign
    campaign_options = {
        f"{c['campaign_name']} ({c['campaign_id']})": c["campaign_id"]
        for c in campaigns
    }
    selected_label = st.selectbox("Selecione uma campanha:", list(campaign_options.keys()))
    selected_campaign_id = campaign_options[selected_label]

    # Analyze button
    if st.button("Analisar Campanha", type="primary"):
        with st.spinner("Carregando dados e fazendo predições..."):
            _analyze_campaign(selected_campaign_id)


def _analyze_campaign(campaign_id: str) -> None:
    """Analyze the selected campaign."""
    # Load campaign data
    try:
        history_df = load_raw_campaign_history(campaign_id)
    except Exception as e:
        st.error(f"Erro ao carregar dados da campanha: {e}")
        return

    if history_df.empty:
        st.warning("Nenhum dado disponível para essa campanha.")
        return

    # Ensure timestamp is datetime
    if "context_timestamp" in history_df.columns:
        history_df["context_timestamp"] = pd.to_datetime(
            history_df["context_timestamp"], errors="coerce", utc=True
        )
        history_df = history_df.sort_values("context_timestamp")

    # Load model
    model = load_model()
    if not model:
        st.error("Modelo não encontrado. Coloque em `/models/v4_lagged/lgbm_fatigue_v4.joblib`.")
        return

    # Compute features
    features_df = compute_lagged_features(history_df)

    # Get feature names from model
    if hasattr(model, "feature_names_"):
        feature_names = list(model.feature_names_)
    else:
        # Try to infer from model attributes
        feature_names = None
        try:
            # For LightGBM
            if hasattr(model, "booster_"):
                feature_names = model.booster_.feature_name()
        except Exception:
            pass

        if not feature_names:
            st.warning(
                "Não foi possível obter os nomes das features do modelo. "
                "Verifique se o modelo foi treinado corretamente."
            )
            return

    # Prepare features for prediction
    try:
        X = prepare_features_for_model(features_df, feature_names)
    except KeyError as e:
        st.error(f"Feature não encontrada nos dados: {e}")
        return

    # Make predictions
    try:
        predictions = model.predict_proba(X)[:, 1]  # Probability of fatigue class
    except Exception as e:
        st.error(f"Erro ao fazer predições: {e}")
        return

    # Add predictions to dataframe
    history_df = history_df.copy()
    history_df["fatigue_prob"] = predictions

    alert_threshold = get_alert_threshold()

    # --- Gauge: Current Risk ---
    st.subheader("Risco Atual")
    current_risk = predictions[-1] if len(predictions) > 0 else 0
    alert_status = "ALERTA" if current_risk >= alert_threshold else "OK"
    alert_color = "red" if current_risk >= alert_threshold else "green"

    fig_gauge = go.Figure(
        data=[
            go.Indicator(
                mode="gauge+number+delta",
                value=current_risk * 100,
                title="Probabilidade de Fadiga (%)",
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": alert_color},
                    "steps": [
                        {"range": [0, alert_threshold * 100], "color": "rgba(0, 255, 0, 0.1)"},
                        {
                            "range": [alert_threshold * 100, 100],
                            "color": "rgba(255, 0, 0, 0.1)",
                        },
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": alert_threshold * 100,
                    },
                },
            )
        ]
    )
    fig_gauge.update_layout(height=300)
    col1, col2 = st.columns([3, 1])
    col1.plotly_chart(fig_gauge, use_container_width=True)
    col2.metric("Status", alert_status, delta=f"{current_risk:.1%}")

    st.markdown("---")

    # --- Timeline: P(fatigue) over time ---
    st.subheader("Linha do Tempo - Probabilidade de Fadiga")

    df_timeline = history_df[["context_timestamp", "fatigue_prob"]].copy()
    df_timeline = df_timeline.dropna()

    if len(df_timeline) > 0:
        fig_timeline = go.Figure()

        # Add line for fatigue probability
        fig_timeline.add_trace(
            go.Scatter(
                x=df_timeline["context_timestamp"],
                y=df_timeline["fatigue_prob"] * 100,
                mode="lines+markers",
                name="P(Fadiga)",
                line=dict(color="blue", width=2),
                marker=dict(size=4),
            )
        )

        # Add threshold line
        fig_timeline.add_hline(
            y=alert_threshold * 100,
            line_dash="dash",
            line_color="red",
            annotation_text="Threshold",
            annotation_position="right",
        )

        # Add shaded band where model alerted
        alerts = df_timeline[df_timeline["fatigue_prob"] >= alert_threshold]
        if len(alerts) > 0:
            # Create bands for alert periods
            fig_timeline.add_vrect(
                x0=alerts["context_timestamp"].iloc[0],
                x1=alerts["context_timestamp"].iloc[-1],
                fillcolor="red",
                opacity=0.1,
                layer="below",
                line_width=0,
            )

        fig_timeline.update_layout(
            title="Probabilidade de Fadiga ao Longo do Tempo",
            xaxis_title="Timestamp",
            yaxis_title="Probabilidade (%)",
            height=400,
            hovermode="x unified",
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("Sem dados de timestamp para exibir a timeline.")

    st.markdown("---")

    # --- Lead Time Analysis ---
    st.subheader("Análise de Lead Time")

    alerts_df = history_df[history_df["fatigue_prob"] >= alert_threshold].copy()
    if len(alerts_df) > 0:
        first_alert_time = alerts_df["context_timestamp"].iloc[0]
        last_timestamp = history_df["context_timestamp"].max()

        # Check if campaign was paused (by looking at spend going to 0)
        spend_candidates = [
            "spend", "campaign_insights_spend_today",
            "campaign_spend", "adAccount_insights_spend_today"
        ]
        spend_col = None
        for col in spend_candidates:
            if col in history_df.columns:
                spend_col = col
                break

        pause_time = None
        if spend_col:
            # Find when spend went to 0 after the first alert
            after_alert = history_df[history_df["context_timestamp"] >= first_alert_time].copy()
            zeros = after_alert[pd.to_numeric(after_alert[spend_col], errors="coerce") == 0]
            if len(zeros) > 0:
                pause_time = zeros["context_timestamp"].iloc[0]

        if pause_time:
            lead_time = (pause_time - first_alert_time).total_seconds() / 3600  # hours
            st.metric(
                "Lead Time",
                f"{lead_time:.1f} horas",
                help="Horas entre o primeiro alerta do modelo e a pausa da campanha",
            )
        else:
            # Campaign might not have been paused
            hours_alerted = (last_timestamp - first_alert_time).total_seconds() / 3600
            st.warning(
                f"Campanha em estado de alerta há {hours_alerted:.1f} horas, "
                "mas não foi pausada."
            )
    else:
        st.info("Nenhum período de alerta detectado nessa campanha.")

    # --- Raw Data ---
    with st.expander("Visualizar dados brutos com predições"):
        display_cols = [
            "context_timestamp",
            "campaign_id",
            "fatigue_prob",
        ]
        display_cols = [c for c in display_cols if c in history_df.columns]
        st.dataframe(
            history_df[display_cols].tail(100),
            use_container_width=True,
        )
