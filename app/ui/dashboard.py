from __future__ import annotations

from typing import Optional
from pathlib import Path
import re

import pandas as pd
import plotly.express as px
import streamlit as st

from app.models.predictor import get_model_info
from app.models.analysis import (
    run_campaign_analysis,
    find_campaign_model,
    find_latest_analysis_file,
)
from app.ui.components import (
    compute_top_drivers,
    plot_feature_importances,
    plot_timeline_with_overlays,
    render_alerts_table,
    render_kpi_row,
)
from app.utils.column_mapper import get_column_mapping_df


def render_schema_page(sample_limit: int = 1000) -> None:
    st.header("Esquema de Dados — Mapeamento de Colunas")
    df = get_column_mapping_df(sample_limit=sample_limit)
    if df.empty:
        st.info("Nenhum arquivo parquet encontrado em data/raw_campaigns.")
        return

    st.write(
        "A tabela abaixo mostra, por coluna, quantos arquivos a incluem (amostra) e os dtypes observados. Espaços vazios são esperados e aguardados na origem dos dados."
    )
    st.dataframe(df, use_container_width=True)
    st.caption("Percentual de valores não-nulos na amostra. Valores ausentes (NaN) são normais e esperados.")


def recommend_action_from_drivers(row: pd.Series) -> tuple[str, float]:
    """Simple heuristic to recommend an action and a confidence score based on drivers."""
    # Basic rules: if budget pressure high -> increase pacing; if cpc acceleration positive -> lower bid
    bp = row.get("budget_pressure_24h")
    cpa = row.get("cpc_acceleration") or row.get("cpc_change")
    risk = float(row.get("fadiga_prevista", 0.0) or 0.0)
    if bp is not None and bp > 0.8:
        return ("Aumentar pacing / verificar budget", min(0.9, 0.3 + risk))
    if cpa is not None and cpa > 0:
        return ("Reduzir bid / revisar criativos", min(0.9, 0.2 + risk))
    if risk >= 0.8:
        return ("Revisar e pausar / reduzir exposição", min(0.95, 0.5 + risk))
    return ("Monitorar — sem ação automática recomendada", max(0.05, risk / 2))


def compute_kpis(enriched_df: pd.DataFrame) -> tuple[dict, list]:
    """Compute KPI metrics with tolerant column name matching.

    Returns (metrics_dict, overlays_list).
    """
    def _norm(s: str) -> str:
        return re.sub(r"[^0-9a-z]", "", str(s).lower())

    def _find_best(df: pd.DataFrame, candidates) -> str | None:
        if df is None or df.columns.empty:
            return None
        col_norm = {c: _norm(c) for c in df.columns}
        for targ in (candidates if isinstance(candidates, (list, tuple)) else [candidates]):
            nt = _norm(targ)
            for c, nc in col_norm.items():
                if nc == nt:
                    return c
        # substring / suffix match
        for targ in (candidates if isinstance(candidates, (list, tuple)) else [candidates]):
            nt = _norm(targ)
            for c, nc in col_norm.items():
                if nc.endswith(nt) or nt in nc or targ.lower() in c.lower():
                    return c
        return None

    # prepare time window (last 24h)
    last_ts = enriched_df["context_timestamp"].dropna().max() if "context_timestamp" in enriched_df.columns else None
    window = pd.DataFrame()
    if last_ts is not None:
        window = enriched_df[enriched_df["context_timestamp"] >= (pd.to_datetime(last_ts) - pd.Timedelta(hours=24))]

    def safe_sum_window(win: pd.DataFrame, candidates) -> float | None:
        if win is None or win.empty:
            return None
        col = _find_best(win, candidates)
        if not col:
            return None
        # if the candidate name indicates 'today' use the last non-null snapshot instead of summing
        if any("today" in str(c).lower() for c in (candidates if isinstance(candidates, (list, tuple)) else [candidates])):
            try:
                last_val = win[col].dropna()
                if not last_val.empty:
                    return float(_to_numeric(last_val).iloc[-1])
            except Exception:
                return None
        try:
            return float(_to_numeric(win[col]).fillna(0.0).sum())
        except Exception:
            return None

    def _to_numeric(series: pd.Series) -> pd.Series:
        # coerce common numeric strings (remove currency symbols, replace comma decimal)
        try:
            s = series.astype(str).str.replace(r"[^0-9,\.\-]", "", regex=True).str.replace(",", ".", regex=False)
            return pd.to_numeric(s, errors="coerce")
        except Exception:
            return pd.to_numeric(series, errors="coerce")

    spend_candidates = ["spend", "spend_today", "campaign_insights_spend_today", "adAccount_insights_spend_today", "campaign_insights_spend_last7d", "adAccount_insights_spend_last7d"]
    spend_today = safe_sum_window(window, spend_candidates)

    daily_budget = None
    db_col = _find_best(enriched_df, ["daily_budget", "campaign_dailyBudget", "campaign_daily_budget", "campaign_budget"])
    if db_col and db_col in enriched_df.columns:
        try:
            last_db = pd.to_numeric(enriched_df[db_col].dropna(), errors="coerce")
            if not last_db.empty:
                daily_budget = float(last_db.iloc[-1])
        except Exception:
            daily_budget = None

    budget_pressure_24h = None
    bp_col = _find_best(window, ["budget_pressure_24h", "budget_pressure", "budget_pressure_24"])
    if bp_col and bp_col in window.columns:
        try:
            budget_pressure_24h = float(_to_numeric(window[bp_col]).mean())
        except Exception:
            budget_pressure_24h = None

    # cpc: either explicit rolling columns or compute from base cpc column
    cpc_3d = None
    cpc_7d = None
    cpc3_col = _find_best(enriched_df, ["cpc_3d", "cpc_3", "cpc_3days"])
    cpc7_col = _find_best(enriched_df, ["cpc_7d", "cpc_7", "cpc_7days"])
    base_cpc_col = _find_best(enriched_df, ["cpc", "avg_cpc", "campaign_cpc", "cpc_avg"])
    if cpc3_col and cpc3_col in enriched_df.columns:
        try:
            cpc_3d = float(pd.to_numeric(enriched_df[cpc3_col], errors="coerce").dropna().iloc[-1])
        except Exception:
            cpc_3d = None
    if cpc7_col and cpc7_col in enriched_df.columns:
        try:
            cpc_7d = float(pd.to_numeric(enriched_df[cpc7_col], errors="coerce").dropna().iloc[-1])
        except Exception:
            cpc_7d = None
    if cpc_3d is None and cpc_7d is None and base_cpc_col:
        try:
            cpc = _to_numeric(enriched_df[base_cpc_col])
            if not cpc.dropna().empty:
                cpc_3d = float(cpc.rolling(3, min_periods=1).mean().iloc[-1])
                cpc_7d = float(cpc.rolling(7, min_periods=1).mean().iloc[-1])
            else:
                cpc_3d = cpc_7d = None
        except Exception:
            cpc_3d = cpc_7d = None

    cpc_acceleration = None
    if cpc_3d is not None and cpc_7d is not None:
        cpc_acceleration = cpc_3d - cpc_7d

    # current risk: prefer 'fadiga_analisada' then 'fadiga_prevista'
    risco_current = None
    for risk_col in ("fadiga_analisada", "fadiga_prevista"):
        if risk_col in enriched_df.columns:
            last_non_null = enriched_df[risk_col].dropna()
            if not last_non_null.empty:
                try:
                    risco_current = float(last_non_null.iloc[-1])
                    break
                except Exception:
                    risco_current = None

    # determine overlays for timeline
    overlays = []
    spend_col_found = _find_best(enriched_df, spend_candidates)
    if spend_col_found:
        overlays.append(spend_col_found)
    if base_cpc_col:
        overlays.append(base_cpc_col)

    # derive budget pressure as spend / daily_budget when missing
    if budget_pressure_24h is None and spend_today is not None and daily_budget:
        try:
            if daily_budget and daily_budget > 0:
                budget_pressure_24h = float(spend_today) / float(daily_budget)
                metrics = {
                    "spend_today": spend_today,
                    "daily_budget": daily_budget,
                    "budget_pressure_24h": budget_pressure_24h,
                    "cpc_3d": cpc_3d,
                    "cpc_7d": cpc_7d,
                    "cpc_acceleration": cpc_acceleration,
                    "risco_fadiga": risco_current,
                }
                return metrics, overlays
        except Exception:
            pass

    metrics = {
        "spend_today": spend_today,
        "daily_budget": daily_budget,
        "budget_pressure_24h": budget_pressure_24h,
        "cpc_3d": cpc_3d,
        "cpc_7d": cpc_7d,
        "cpc_acceleration": cpc_acceleration,
        "risco_fadiga": risco_current,
    }
    return metrics, overlays


def render_overview(enriched_df: pd.DataFrame, model_info: Optional[dict] = None, campaign_id: Optional[str] = None) -> None:
    st.header("Overview — KPI da Campanha")
    if model_info is None:
        model_info = get_model_info()

    alert_threshold = float(model_info.get("alert_threshold", 0.5))

    # Advanced Analysis button at the top for quick access
    st.subheader("Análise Avançada")
    st.write(
        "Execute uma análise local usando os artefatos do modelo (xgb_alert_model.joblib, label_encoders.joblib, model_summary.json)."
    )
    if campaign_id is None:
        st.info("Informe a campanha na barra lateral para permitir a Análise Avançada.")
    else:
        # attempt to find persisted analysis on disk (latest parquet/csv)
        cache_key = f"analysis_parquet_{campaign_id}"
        disk_path = find_latest_analysis_file(campaign_id)
        cached_path = st.session_state.get(cache_key) if hasattr(st, "session_state") else None
        # prefer session cache if it exists and the file is present
        chosen = None
        if cached_path and Path(cached_path).exists():
            chosen = cached_path
        elif disk_path:
            chosen = disk_path

        if chosen:
            try:
                if chosen.endswith('.parquet'):
                    loaded = pd.read_parquet(chosen)
                else:
                    loaded = pd.read_csv(chosen)
                enriched_df = loaded
                st.info(f"Usando resultados previamente gerados em: {chosen}")
                try:
                    st.session_state[cache_key] = chosen
                except Exception:
                    pass
            except Exception:
                # ignore and show button
                pass
        if st.button("Fazer Análise Avançada", key=f"fazer_analise_top_{campaign_id}"):
            with st.spinner("Executando análise e salvando resultados..."):
                result = run_campaign_analysis(enriched_df, campaign_id)
            if result.get("error"):
                st.error(result.get("error"))
            else:
                st.success("Análise concluída e salva.")
                if result.get("parquet"):
                    st.write("Parquet:", result.get("parquet"))
                    try:
                        loaded = pd.read_parquet(result.get("parquet"))
                        st.subheader("Amostra dos resultados gerados")
                        st.dataframe(loaded.head(20), use_container_width=True)
                        # cache path in session for quicker reloads
                        try:
                            st.session_state[cache_key] = result.get("parquet")
                        except Exception:
                            pass
                        # prefer the saved analysis as the dataset for the remaining UI
                        enriched_df = loaded
                    except Exception as e:
                        st.warning(f"Falha ao ler Parquet gerado: {e}")
                else:
                    st.write("CSV:", result.get("csv"))
                    try:
                        st.session_state[cache_key] = result.get("csv")
                    except Exception:
                        pass

    if enriched_df is None or enriched_df.empty:
        st.info("Sem histórico para a campanha selecionada.")
        return

    # Ensure timestamp and copy dataset
    if "context_timestamp" in enriched_df.columns:
        enriched_df = enriched_df.sort_values("context_timestamp").copy()
        enriched_df["context_timestamp"] = pd.to_datetime(enriched_df["context_timestamp"], errors="coerce")
    else:
        enriched_df = enriched_df.copy()

    # KPIs: compute using tolerant column matching
    metrics, overlays = compute_kpis(enriched_df)
    render_kpi_row(metrics, alert_threshold=alert_threshold)

    st.markdown("---")

    # Timeline — show risk as a line chart (timestamp vs risk)
    st.subheader("Linha do Tempo — Risco")
    # prefer analisada then prevista
    risk_col = "fadiga_analisada" if "fadiga_analisada" in enriched_df.columns else "fadiga_prevista"
    if "context_timestamp" in enriched_df.columns and risk_col in enriched_df.columns:
        df_risk = enriched_df[["context_timestamp", risk_col]].copy()
        df_risk["context_timestamp"] = pd.to_datetime(df_risk["context_timestamp"], errors="coerce")
        df_risk = df_risk.sort_values("context_timestamp").reset_index(drop=True)
        # coerce numeric and drop NaNs for plotting
        df_risk[risk_col] = pd.to_numeric(df_risk[risk_col], errors="coerce")
        vals = df_risk[risk_col].dropna()
        if vals.empty or vals.nunique() <= 1:
            st.info(f"As previsões em `{risk_col}` são constantes ou ausentes — verifique avisos abaixo.")
        else:
            fig = px.line(df_risk, x="context_timestamp", y=risk_col, markers=True, title="Risco de Fadiga ao longo do tempo")
            fig.add_hline(y=alert_threshold, line_dash="dash", line_color="#ffa600")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sem dados de tempo/risco para exibir o gráfico de risco.")

    st.markdown("---")

    # Separate visualization for Spend / CPC to avoid cluttering the risk timeline
    st.subheader("Spend / CPC (separado)")
    if overlays:
        # copy and mask midnight 'today' zeros which are artifacts of daily resets
        df_overlays = enriched_df.copy()
        if "context_timestamp" in df_overlays.columns:
            df_overlays["context_timestamp"] = pd.to_datetime(df_overlays["context_timestamp"], errors="coerce")
            try:
                for col in overlays:
                    if col in df_overlays.columns:
                        # mask zeros recorded at midnight for 'today' style columns
                        if "today" in col.lower() or "spend" in col.lower():
                            mask = df_overlays["context_timestamp"].dt.hour == 0
                            df_overlays.loc[mask, col] = pd.NA
            except Exception:
                pass

        # if there are overlay columns present, plot them without risk on primary axis
        present = [c for c in overlays if c in df_overlays.columns]
        if present:
            plot_timeline_with_overlays(
                df_overlays,
                time_col="context_timestamp",
                risk_col="__NO_RISK_COLUMN__",
                overlays=present,
                alert_threshold=alert_threshold,
                title="Spend / CPC ao longo do tempo",
            )
        else:
            st.info("Nenhuma coluna de spend/cpc disponível para visualização separada.")
    else:
        st.info("Nenhuma coluna de spend/cpc identificada para a campanha.")

    st.markdown("---")

    # Alerts / Feed
    st.subheader("Alertas / Feed")
    st.write("Mostra snapshots com maior risco ou que ultrapassaram o limiar configurado. Útil para priorizar ações rápidas.")
    render_alerts_table(enriched_df, alert_threshold=alert_threshold, risk_col_preference=[risk_col])

    st.markdown("---")

    # Driver Analysis
    st.subheader("Driver Analysis — Top features")
    # correlate numeric features with the active risk column
    if risk_col in enriched_df.columns:
        numeric = enriched_df.select_dtypes(include=["number"]).copy()
        if risk_col in numeric.columns:
            # drop the risk column itself for candidate drivers
            corr = numeric.corr()[risk_col].abs().sort_values(ascending=False)
            corr = corr.drop(labels=[risk_col]) if risk_col in corr.index else corr
            # drop constant features
            corr = corr[corr.index.to_series().apply(lambda c: numeric[c].nunique() > 1)]
            if corr.empty:
                st.info("Não há variação útil nas features numéricas para identificar drivers.")
            else:
                corr_df = corr.reset_index()
                corr_df.columns = ["feature", "abs_corr_with_risk"]
                fig = px.bar(corr_df.head(20), x="abs_corr_with_risk", y="feature", orientation="h", title="Features correlacionadas com risco (|corr|)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Coluna de risco `{risk_col}` presente, mas não há features numéricas para correlacionar.")
    else:
        st.info(f"Coluna de risco `{risk_col}` não encontrada para análise de drivers locais.")

    st.markdown("---")

    # Model health
    st.subheader("Performance / Saúde do Modelo")
    st.write("Modelo: ", model_info.get("model_version") if model_info else "(não disponível)")
    st.write("Alert threshold: ", float(model_info.get("alert_threshold", 0.5)))
    if model_info and model_info.get("metrics_holdout_auc"):
        st.write("Holdout AUC:", model_info.get("metrics_holdout_auc"))

    # show distribution of active risk column
    if risk_col in enriched_df.columns:
        vals = pd.to_numeric(enriched_df[risk_col], errors="coerce").dropna()
        if vals.empty or vals.nunique() <= 1:
            st.info(f"As previsões em `{risk_col}` são constantes ou ausentes — verifique avisos abaixo.")
            # surface warnings if present
            if hasattr(enriched_df, "attrs"):
                if enriched_df.attrs.get("prediction_error"):
                    st.error("Erro de predição: " + str(enriched_df.attrs.get("prediction_error")))
                if enriched_df.attrs.get("prediction_warnings"):
                    for w in enriched_df.attrs.get("prediction_warnings"):
                        st.warning(w)
        else:
            st.write("Distribuição de probabilidades previstas")
            fig = px.histogram(enriched_df, x=risk_col, nbins=40, title=f"Distribuição de {risk_col}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Coluna de risco `{risk_col}` não encontrada no dataset.")

    st.markdown("---")

    # Segmentation / Aggregates
    st.subheader("Segmentação / Agregados")
    # choose a segmentation column with more than one unique value
    candidate_seg = [
        c
        for c in ["adset_id", "adset_name", "placement", "campaign_category", "campaign_objective"]
        if c in enriched_df.columns and enriched_df[c].nunique() > 1
    ]
    if candidate_seg:
        seg = candidate_seg[0]
        grouping = (
            enriched_df.groupby(seg)[risk_col].agg(["mean", "count"]).reset_index()
        )
        fig = px.bar(grouping, x=seg, y=["mean", "count"], title=f"Risco médio e volume por {seg}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        # fallback: time-based daily aggregation
        if "context_timestamp" in enriched_df.columns:
            try:
                daily = enriched_df.set_index("context_timestamp")[risk_col].resample("D").agg(["mean", "count"]).reset_index()
                fig = px.line(daily, x="context_timestamp", y="mean", title="Risco médio diário")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Segmentação não disponível para esse dataset.")
        else:
            st.info("Segmentação não disponível para esse dataset.")

    st.markdown("---")

    # Action recommendation
    st.subheader("Ação Recomendada")
    # prefer analisada then prevista
    if risk_col in enriched_df.columns:
        non_null_idx = enriched_df[risk_col].dropna().index
        if len(non_null_idx) > 0:
            last_row = enriched_df.loc[non_null_idx[-1]].copy()
            # normalize for recommendation function which expects 'fadiga_prevista'
            if risk_col != "fadiga_prevista":
                last_row["fadiga_prevista"] = last_row.get(risk_col)
            suggestion, confidence = recommend_action_from_drivers(last_row)
            st.write(f"Sugestão: **{suggestion}** — confiança: {confidence:.2f}")
        else:
            st.info("Sem previsões disponíveis para gerar recomendações.")
        st.write("Ações automatizadas exigem confirmação humana. Use 'Marcar ação' para registrar a intenção.")
        if st.button("Marcar ação como 'Revisar'", key="mark_action"):
            st.success("Ação registrada — revisar manualmente. (Não foi aplicada automaticamente.)")
    else:
        st.info("Sem coluna de risco disponível para gerar recomendações.")

    st.caption("Observação: valores ausentes (NaN) são esperados nos dados brutos e não impedem a análise; o modelo ou as heurísticas tratam NaNs onde aplicável.")
