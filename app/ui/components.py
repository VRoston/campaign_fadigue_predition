from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_kpi_row(metrics: dict, alert_threshold: float = 0.5) -> None:
    """Render a row of KPI cards from a mapping name -> value.

    The function treats any key containing 'risco' as the risk metric and
    annotates it with the alert threshold.
    """
    if not metrics:
        return
    cols = st.columns(len(metrics))
    for (label, value), col in zip(metrics.items(), cols):
        display_label = label.replace("_", " ").title()
        if isinstance(value, (int, float)) and not math.isnan(value):
            if abs(value) >= 1000:
                disp = f"{value:,.0f}"
            else:
                disp = f"{value:,.2f}"
        else:
            disp = "—" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)

        # Flag risk card visually via metric help text
        if "risco" in label.lower() or label.lower().startswith("risco"):
            is_alert = False
            try:
                is_alert = float(value) >= float(alert_threshold)
            except Exception:
                is_alert = False
            help_text = f"ALERTA se ≥ {alert_threshold:.2f}" if is_alert else f"Limiar: {alert_threshold:.2f}"
            col.metric(display_label, disp, help=help_text)
        else:
            col.metric(display_label, disp)


def plot_timeline_with_overlays(
    df: pd.DataFrame,
    time_col: str = "context_timestamp",
    risk_col: str = "fadiga_prevista",
    overlays: Iterable[str] | None = None,
    alert_threshold: float = 0.5,
    title: str | None = None,
) -> None:
    if overlays is None:
        overlays = ["spend", "cpc"]
    if time_col not in df.columns:
        st.info("A coluna de tempo não está disponível para o timeline.")
        return

    df_plot = df.sort_values(time_col).copy()
    fig = go.Figure()

    if risk_col in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_plot[time_col],
                y=df_plot[risk_col],
                mode="lines+markers",
                name="Risco Fadiga",
                line=dict(color="#e45756"),
            )
        )
        fig.add_hline(y=alert_threshold, line_dash="dash", line_color="#ffa600")

        # Add shaded vrects for contiguous ranges above threshold
        above = df_plot[df_plot[risk_col] >= alert_threshold]
        if not above.empty:
            intervals = []
            last = None
            start = None
            for t in pd.to_datetime(above[time_col]):
                if last is None:
                    start = t
                    last = t
                elif (t - last).total_seconds() > 3600 * 24:
                    intervals.append((start, last))
                    start = t
                    last = t
                else:
                    last = t
            if start is not None and last is not None:
                intervals.append((start, last))

            for s, e in intervals:
                fig.add_vrect(x0=s, x1=e, fillcolor="#e45756", opacity=0.08, layer="below", line_width=0)

    # overlays on secondary y axis
    overlay_present = any(c in df_plot.columns for c in overlays)
    if overlay_present:
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))
        for overlay in overlays:
            if overlay in df_plot.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_plot[time_col],
                        y=df_plot[overlay],
                        mode="lines",
                        name=overlay,
                        yaxis="y2",
                    )
                )

    fig.update_layout(title=title or "Timeline", xaxis_title="timestamp")
    st.plotly_chart(fig, use_container_width=True)


def compute_top_drivers(df: pd.DataFrame, exclude_cols: list | None = None, n: int = 3) -> pd.DataFrame:
    """Estimate top-n drivers per row using absolute z-score vs median (sample-aware).

    This is a lightweight heuristic useful when SHAP or model-local explanations
    are not available. Returns a DataFrame with `top_drivers` (list) and `top_values`.
    """
    if exclude_cols is None:
        exclude_cols = ["fadiga_prevista", "fadiga_analisada", "context_timestamp"]
    numeric = df.select_dtypes(include=["number"]).copy()
    for c in exclude_cols:
        if c in numeric.columns:
            numeric = numeric.drop(columns=[c])
    if numeric.empty:
        return pd.DataFrame(columns=["top_drivers", "top_values"])

    med = numeric.median()
    # replace zeros with NaN safely (preserve float dtype), then fill
    std = numeric.std().replace(0, np.nan).fillna(1.0)
    z = (numeric - med) / std
    z = z.abs().fillna(0.0)

    top_drivers = z.apply(lambda row: row.nlargest(n).index.tolist(), axis=1)
    top_values = z.apply(lambda row: row.nlargest(n).values.tolist(), axis=1)
    return pd.DataFrame({"top_drivers": top_drivers, "top_values": top_values})


def render_alerts_table(
    df: pd.DataFrame,
    alert_threshold: float = 0.5,
    max_rows: int = 20,
    risk_col_preference: list | None = None,
) -> None:
    if risk_col_preference is None:
        risk_col_preference = ["fadiga_analisada", "fadiga_prevista"]

    risk_col = None
    for cand in risk_col_preference:
        if cand in df.columns:
            risk_col = cand
            break

    if risk_col is None:
        st.info("Sem previsões de fadiga para listar alertas.")
        return

    alert_rows = df[df[risk_col] >= alert_threshold].copy()
    if alert_rows.empty:
        # show top N by risk
        alert_rows = df.sort_values(risk_col, ascending=False).head(max_rows).copy()
    else:
        alert_rows = alert_rows.sort_values(risk_col, ascending=False).head(max_rows)

    drivers_df = compute_top_drivers(alert_rows)
    drivers_col = []
    for idx, row in drivers_df.iterrows():
        names = row.get("top_drivers", [])
        vals = row.get("top_values", [])
        parts = []
        for n, v in zip(names, vals):
            parts.append(f"{n}: {v:.2f}")
        drivers_col.append("; ".join(parts))

    display_df = alert_rows.copy()
    display_df["top_drivers"] = drivers_col
    # show limited columns for readability
    cols = [c for c in ["context_timestamp", risk_col, "top_drivers"] if c in display_df.columns]
    st.dataframe(display_df[cols], use_container_width=True)


def plot_feature_importances(names, importances, top_n: int = 20):
    if not names or importances is None:
        st.info("No feature importance data available.")
        return
    import pandas as _pd

    df = _pd.DataFrame({"feature": names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n)
    fig = px.bar(df, x="importance", y="feature", orientation="h", title="Top Features")
    st.plotly_chart(fig, use_container_width=True)
