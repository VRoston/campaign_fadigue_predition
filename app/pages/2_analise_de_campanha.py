"""Campaign Analysis page - analyzes individual campaigns."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from app.config import (
    MODEL,
    THRESHOLD,
    FEATURE_COLS,
    FEATURE_CATEGORIES,
    FEATURE_DISPLAY_NAMES,
)
from app.features import compute_features_72h, prepare_features_for_model
from app.duckdb_utils import (
    get_ad_accounts,
    get_campaigns_by_account,
    get_campaign_snapshots,
)


@st.cache_data(ttl=300, show_spinner=False)
def calculate_probability_over_time(df_campaign: pd.DataFrame) -> pd.DataFrame | None:
    """Calculate model probability at each snapshot (cached per campaign DataFrame)."""
    if df_campaign is None or len(df_campaign) < 3:
        return None

    data = {"timestamp": [], "age_hours": [], "probability": []}

    for idx in range(len(df_campaign)):
        df_slice = df_campaign.iloc[:idx + 1]
        try:
            features = compute_features_72h(df_slice)
            if len(features) > 0:
                X = prepare_features_for_model(features.to_dict(), FEATURE_COLS)
                prob = MODEL.predict_proba(X)[0, 1]
                data["probability"].append(prob)
                data["timestamp"].append(df_slice["context_timestamp"].iloc[-1])
                data["age_hours"].append(df_slice["age_hours"].iloc[-1])
        except Exception:
            pass

    if len(data["probability"]) < 3:
        return None

    return pd.DataFrame(data)


# Page title
st.markdown("# 🎯 Análise de Campanha")
st.markdown("---")

st.markdown("### Seleção de Campanha")

# Initialize session state
if "selected_account" not in st.session_state:
    st.session_state.selected_account = None
if "selected_campaign" not in st.session_state:
    st.session_state.selected_campaign = None
if "campaigns_cache" not in st.session_state:
    st.session_state.campaigns_cache = {}

# Step 1: Load Ad Accounts
with st.spinner("Carregando Ad Accounts..."):
    try:
        accounts_dict = get_ad_accounts()
    except Exception as e:
        st.error(f"❌ Erro ao carregar Ad Accounts: {e}")
        accounts_dict = {}

if not accounts_dict:
    st.error("❌ Nenhuma Ad Account encontrada. Verifique CAMPAIGN_DATA_PATH.")
    st.stop()

# Account selection
col_account, col_campaign, col_button = st.columns([2, 2, 1])

with col_account:
    account_ids = sorted(accounts_dict.keys())
    account_options = [f"{aid} - {accounts_dict[aid]}" for aid in account_ids]

    selected_idx = st.selectbox(
        "📊 Ad Account",
        range(len(account_options)),
        format_func=lambda i: account_options[i],
        key="account_select",
        help="Selecione uma conta de anúncios"
    )
    selected_account = account_ids[selected_idx]

    if selected_account != st.session_state.selected_account:
        st.session_state.selected_account = selected_account
        st.session_state.selected_campaign = None
        st.session_state.campaigns_cache = {}

# Step 2: Load Campaigns for selected account (on demand)
campaigns_dict = {}
if selected_account:
    if selected_account not in st.session_state.campaigns_cache:
        with st.spinner(f"Carregando campanhas de {selected_account}..."):
            try:
                campaigns_dict = get_campaigns_by_account(selected_account)
                st.session_state.campaigns_cache[selected_account] = campaigns_dict
            except Exception as e:
                st.error(f"❌ Erro ao carregar campanhas: {e}")
                campaigns_dict = {}
    else:
        campaigns_dict = st.session_state.campaigns_cache[selected_account]

with col_campaign:
    if not campaigns_dict:
        st.warning("Nenhuma campanha nesta conta")
        st.stop()

    campaign_ids_list = sorted(campaigns_dict.keys())
    campaign_options = [
        f"{cid} - {campaigns_dict.get(cid, cid)}"
        for cid in campaign_ids_list
    ]

    selected_idx = st.selectbox(
        "📌 Campanha",
        range(len(campaign_options)),
        format_func=lambda i: campaign_options[i],
        key="campaign_select",
    )
    selected_campaign = campaign_ids_list[selected_idx]

with col_button:
    st.write("")
    st.write("")
    analyze_button = st.button("🔍 Analisar", use_container_width=True, type="primary")

st.markdown("---")

if not analyze_button:
    st.info("👈 Selecione uma campanha e clique em 'Analisar'")
    st.stop()

# Step 3: Load full campaign data
with st.spinner(f"Carregando dados da campanha {selected_campaign}..."):
    try:
        df_raw = get_campaign_snapshots(selected_campaign)
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
        st.stop()

if df_raw is None or len(df_raw) == 0:
    st.error(f"❌ Nenhum dado encontrado para campanha {selected_campaign}")
    st.stop()

# Parse timestamps
df_raw["context_timestamp"] = pd.to_datetime(
    df_raw["context_timestamp"], utc=True, errors="coerce"
)
df_raw["campaign_startTime"] = pd.to_datetime(
    df_raw["campaign_startTime"], utc=True, errors="coerce"
)

df_raw = df_raw.dropna(subset=["context_timestamp"])

if len(df_raw) == 0:
    st.error("❌ Nenhum dado válido encontrado (context_timestamp inválido em todos os registros)")
    st.stop()

# Infer start time from first snapshot if not set
if df_raw["campaign_startTime"].isna().all():
    df_raw["campaign_startTime"] = df_raw["context_timestamp"].min()
df_raw["campaign_startTime"] = df_raw["campaign_startTime"].fillna(
    df_raw["context_timestamp"].min()
)

# Calculate age from campaign start
df_raw["age_hours"] = (
    (df_raw["context_timestamp"] - df_raw["campaign_startTime"]).dt.total_seconds() / 3600
)

# Filter to first 72h of ACTIVE status
df_campaign = df_raw[
    (df_raw["age_hours"] >= 0)
    & (df_raw["age_hours"] <= 72)
    & (df_raw["campaign_status"] == "ACTIVE")
].copy()

if len(df_campaign) == 0:
    total_snaps = len(df_raw[(df_raw["age_hours"] >= 0) & (df_raw["age_hours"] <= 72)])
    if total_snaps > 0:
        statuses = df_raw[
            (df_raw["age_hours"] >= 0) & (df_raw["age_hours"] <= 72)
        ]["campaign_status"].value_counts().to_dict()
        st.error(
            f"❌ Nenhuma atividade ACTIVE nos primeiros 72h\n\n"
            f"Status encontrados: {statuses}"
        )
    else:
        st.error("❌ Nenhum snapshot nas primeiras 72h encontrado")
    st.stop()

if len(df_campaign) < 3:
    st.warning(
        f"⚠️ **Dados insuficientes**\n\n"
        f"Snapshots ACTIVE nas 72h: {len(df_campaign)} (mínimo: 3)"
    )
    st.stop()

df_campaign = df_campaign.sort_values("context_timestamp").reset_index(drop=True)

# Calculate final prediction
with st.spinner("Calculando previsão..."):
    try:
        features = compute_features_72h(df_campaign)
        if len(features) == 0:
            st.error("❌ Não foi possível calcular as features para esta campanha")
            st.stop()
        X = prepare_features_for_model(features.to_dict(), FEATURE_COLS)
        probability = MODEL.predict_proba(X)[0, 1]
        prediction = probability >= THRESHOLD
    except Exception as e:
        st.error(f"❌ Erro ao calcular previsão: {e}")
        st.stop()

# Result card
if prediction:
    result_color = "#1e7e34"
    icon = "✓"
    result_text = "Campanha com Potencial"
    subtext = "Indicadores sugerem sucesso"
else:
    result_color = "#b91c1c"
    icon = "✗"
    result_text = "Campanha em Risco"
    subtext = "Indicadores alertam para dificuldades"

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, {result_color} 0%, {result_color}dd 100%);
    padding: 30px;
    border-radius: 10px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
">
    <div style="font-size: 20px; margin-bottom: 8px; font-weight: 500;">{icon} {result_text}</div>
    <div style="font-size: 14px; margin-bottom: 20px; opacity: 0.95;">{subtext}</div>
    <div style="font-size: 48px; font-weight: bold;">{probability*100:.1f}%</div>
    <div style="font-size: 12px; margin-top: 8px; opacity: 0.8;">Threshold: {THRESHOLD*100:.1f}% | Snapshots: {len(df_campaign)}</div>
</div>
""", unsafe_allow_html=True)

# Metrics
st.markdown("### Métricas Principais")
m1, m2, m3 = st.columns(3, gap="medium")

with m1:
    gasto = features.get("spend_72h", np.nan)
    if not np.isnan(gasto):
        st.metric("💰 Gasto 72h", f"R$ {gasto:,.0f}")
    else:
        st.metric("💰 Gasto 72h", "—")

with m2:
    sb = features.get("spend_vs_budget", np.nan)
    if not np.isnan(sb):
        st.metric("📊 Spend vs Budget", f"{sb*100:.0f}%")
    else:
        st.metric("📊 Spend vs Budget", "—")

with m3:
    cpc = features.get("cpc_72h", np.nan)
    if not np.isnan(cpc):
        st.metric("💵 CPC Médio", f"R$ {cpc:.2f}")
    else:
        st.metric("💵 CPC Médio", "—")

st.markdown("---")

# Evolution chart
st.markdown("### Evolução do Risco (72h)")

with st.spinner("Calculando evolução temporal..."):
    prob_over_time = calculate_probability_over_time(df_campaign)

if prob_over_time is not None and len(prob_over_time) > 2:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=prob_over_time["age_hours"],
        y=prob_over_time["probability"],
        mode="lines",
        name="Probabilidade",
        line=dict(color="#3b82f6", width=3),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.2)",
    ))

    fig.add_hline(
        y=THRESHOLD,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Threshold ({THRESHOLD*100:.1f}%)",
        annotation_position="right",
    )

    fig.update_layout(
        title="Progressão da probabilidade ao longo das 72h",
        xaxis_title="Idade (horas)",
        yaxis_title="Probabilidade",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=450,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("ℹ️ Dados insuficientes para gráfico de evolução temporal")

st.markdown("---")

# Features table
with st.expander("📊 **Ver Detalhes das 26 Métricas**", expanded=False):
    st.markdown("Valores de cada feature calculada pelo modelo:")

    nan_count = sum(
        1 for feat in FEATURE_COLS
        if isinstance(features.get(feat, np.nan), float) and np.isnan(features.get(feat, np.nan))
    )
    if nan_count > 0:
        st.warning(f"⚠️ {nan_count} features com valor ausente (NaN) — o modelo usa média do treino como fallback")

    for category, feature_list in FEATURE_CATEGORIES.items():
        st.markdown(f"#### {category}")

        rows = []
        for feat in feature_list:
            val = features.get(feat, np.nan)
            name = FEATURE_DISPLAY_NAMES.get(feat, feat)
            is_nan = isinstance(val, float) and np.isnan(val)

            if is_nan:
                rows.append({"Métrica": name, "Valor": "—", "Status": "⚠️"})
            else:
                if isinstance(val, float):
                    formatted = f"{val:.4f}" if abs(val) < 10 else f"{val:.2f}"
                else:
                    formatted = str(val)
                rows.append({"Métrica": name, "Valor": formatted, "Status": "✓"})

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
