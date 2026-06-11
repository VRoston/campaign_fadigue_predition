from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw_campaigns"


def _parquet_source() -> str:
    """
    Returns a path pattern for DuckDB's read_parquet.

    Respects the environment variable `CAMPAIGN_DATA_PATH` which may be:
      - an absolute directory (e.g. /home/vroston/data/raw_campaigns)
      - an absolute glob pattern (e.g. /home/vroston/data/raw_campaigns/*.parquet)
      - a single parquet file path
    If not set, falls back to the repository `data/raw_campaigns/*.parquet`.
    """
    env = os.environ.get("CAMPAIGN_DATA_PATH")
    if env:
        env_str = str(env)
        if "*" in env_str or env_str.endswith(".parquet"):
            return env_str
        return (Path(env_str).expanduser().resolve() / "*.parquet").as_posix()
    return (DEFAULT_RAW_DIR / "*.parquet").as_posix()


def _run_query(query: str, params: tuple | None = None) -> pd.DataFrame:
    params = params or ()
    source = _parquet_source()
    parent = Path(source).parent
    # quick existence check: make sure there is at least one parquet file
    if not any(parent.glob("*.parquet")):
        return pd.DataFrame()

    with duckdb.connect(database=":memory:") as conn:
        conn.execute("PRAGMA disable_progress_bar")
        try:
            return conn.execute(query, [source, *params]).df()
        except duckdb.Error as exc:
            raise RuntimeError(f"DuckDB query failed: {exc}") from exc


def get_all_ad_accounts() -> pd.DataFrame:
    query = """
        SELECT DISTINCT
            CAST(adAccount_id AS VARCHAR) AS adAccount_id,
            CAST(adAccount_name AS VARCHAR) AS adAccount_name
        FROM read_parquet(?, union_by_name=true)
        WHERE adAccount_id IS NOT NULL
        ORDER BY adAccount_name, adAccount_id
    """
    return _run_query(query)


def get_campaigns_by_account(account_id: str) -> pd.DataFrame:
    query = """
        SELECT DISTINCT
            CAST(campaign_id AS VARCHAR) AS campaign_id,
            CAST(campaign_name AS VARCHAR) AS campaign_name
        FROM read_parquet(?, union_by_name=true)
        WHERE CAST(adAccount_id AS VARCHAR) = ?
          AND campaign_id IS NOT NULL
        ORDER BY campaign_name, campaign_id
    """
    return _run_query(query, (str(account_id),))


def get_all_campaigns() -> pd.DataFrame:
    query = """
        SELECT DISTINCT
            CAST(campaign_id AS VARCHAR) AS campaign_id,
            CAST(campaign_name AS VARCHAR) AS campaign_name
        FROM read_parquet(?, union_by_name=true)
        WHERE campaign_id IS NOT NULL
        ORDER BY campaign_name, campaign_id
    """
    return _run_query(query)


def get_batch_campaign_history(n_campaigns: int = 200, rows_per_campaign: int = 48) -> pd.DataFrame:
    """
    Returns the last `rows_per_campaign` snapshots for the top-N most-active campaigns.
    Used to build a cross-sectional reference distribution of risk scores.
    """
    source = _parquet_source()
    parent = Path(source).parent
    if not any(parent.glob("*.parquet")):
        return pd.DataFrame()

    # Inline source in query — path comes from env var, not user input
    query = f"""
        WITH top_campaigns AS (
            SELECT CAST(campaign_id AS VARCHAR) AS cid
            FROM read_parquet('{source}', union_by_name=true)
            GROUP BY 1
            ORDER BY COUNT(*) DESC
            LIMIT {n_campaigns}
        )
        SELECT s.*
        FROM read_parquet('{source}', union_by_name=true) s
        INNER JOIN top_campaigns tc ON CAST(s.campaign_id AS VARCHAR) = tc.cid
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY CAST(s.campaign_id AS VARCHAR)
            ORDER BY s.context_timestamp DESC
        ) <= {rows_per_campaign}
        ORDER BY CAST(s.campaign_id AS VARCHAR), s.context_timestamp
    """
    with duckdb.connect(database=":memory:") as conn:
        conn.execute("PRAGMA disable_progress_bar")
        try:
            return conn.execute(query).df()
        except duckdb.Error as exc:
            raise RuntimeError(f"DuckDB query failed: {exc}") from exc


def load_raw_campaign_history(campaign_id: str) -> pd.DataFrame:
    query = """
        SELECT *
        FROM read_parquet(?, union_by_name=true)
        WHERE CAST(campaign_id AS VARCHAR) = ?
        ORDER BY context_timestamp
    """
    return _run_query(query, (str(campaign_id),))
