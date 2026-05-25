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


def load_raw_campaign_history(campaign_id: str) -> pd.DataFrame:
    query = """
        SELECT *
        FROM read_parquet(?, union_by_name=true)
        WHERE CAST(campaign_id AS VARCHAR) = ?
        ORDER BY context_timestamp
    """
    return _run_query(query, (str(campaign_id),))
