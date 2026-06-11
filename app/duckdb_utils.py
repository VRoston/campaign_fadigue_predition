"""DuckDB utilities for efficient data loading."""

import duckdb
import streamlit as st
from pathlib import Path
from app.config import DATA_PATH

_PARQUET_GLOB = str(DATA_PATH / "*.parquet")


def _connect() -> duckdb.DuckDBPyConnection:
    """Create a fresh in-memory DuckDB connection (cheap, thread-safe)."""
    return duckdb.connect(":memory:")


@st.cache_data(ttl=300, show_spinner=False)
def get_ad_accounts() -> dict:
    """Get unique ad accounts with names (cached 5 min)."""
    conn = _connect()
    query = f"""
    SELECT DISTINCT
        CAST(adAccount_id AS VARCHAR) AS account_id,
        CAST(adAccount_name AS VARCHAR) AS account_name
    FROM read_parquet('{_PARQUET_GLOB}', union_by_name=true)
    WHERE adAccount_id IS NOT NULL
    ORDER BY account_id
    """
    try:
        result = conn.execute(query).fetchall()
        return {
            str(aid): str(aname) if aname else str(aid)
            for aid, aname in result
            if aid
        }
    except Exception as e:
        print(f"Error loading ad accounts: {e}")
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def get_campaigns_by_account(account_id: str) -> dict:
    """Get campaigns for a specific account (cached 5 min)."""
    conn = _connect()
    query = f"""
    SELECT DISTINCT
        CAST(campaign_id AS VARCHAR) AS campaign_id,
        CAST(campaign_name AS VARCHAR) AS campaign_name
    FROM read_parquet('{_PARQUET_GLOB}', union_by_name=true)
    WHERE CAST(adAccount_id AS VARCHAR) = $1
      AND campaign_id IS NOT NULL
    ORDER BY campaign_id
    """
    try:
        result = conn.execute(query, [account_id]).fetchall()
        return {str(cid): str(cname) if cname else str(cid) for cid, cname in result}
    except Exception as e:
        print(f"Error loading campaigns for {account_id}: {e}")
        return {}


@st.cache_data(ttl=60, show_spinner=False)
def get_campaign_snapshots(campaign_id: str):
    """Get all snapshots for a campaign, filtered to first 72h (cached 1 min)."""
    conn = _connect()
    query = f"""
    SELECT *
    FROM read_parquet('{_PARQUET_GLOB}', union_by_name=true)
    WHERE CAST(campaign_id AS VARCHAR) = $1
    ORDER BY context_timestamp
    """
    try:
        df = conn.execute(query, [campaign_id]).fetch_df()
        return df if len(df) > 0 else None
    except Exception as e:
        print(f"Error loading campaign snapshots for {campaign_id}: {e}")
        return None
