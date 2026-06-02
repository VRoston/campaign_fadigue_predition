from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw_campaigns"


def _parquet_source() -> str:
    """Resolve the parquet source pattern honoring CAMPAIGN_DATA_PATH env var."""
    env = None
    try:
        import os

        env = os.environ.get("CAMPAIGN_DATA_PATH")
    except Exception:
        env = None

    if env:
        env_str = str(env)
        if "*" in env_str or env_str.endswith(".parquet"):
            return env_str
        return (Path(env_str).expanduser().resolve() / "*.parquet").as_posix()
    return (DEFAULT_RAW_DIR / "*.parquet").as_posix()


def scan_parquet_schema(sample_limit: int = 1000) -> pd.DataFrame:
    """Scans available parquet files and returns a table with column presence and dtypes.

    The returned DataFrame has columns: file_path, column, dtype, non_null_count
    and is useful to build an overview of which files contain which columns. Uses a
    sampled read to avoid loading all rows.
    """
    source = _parquet_source()
    parent = Path(source).parent
    if not any(parent.glob("*.parquet")):
        return pd.DataFrame()

    rows: List[Dict] = []
    with duckdb.connect(database=":memory:") as conn:
        conn.execute("PRAGMA disable_progress_bar")
        files = list(parent.glob("*.parquet"))
        for p in files:
            try:
                # get columns (fast) and a small sample for dtype/counts
                sample = conn.execute(
                    "SELECT * FROM read_parquet(?) LIMIT ?", [p.as_posix(), int(sample_limit)]
                ).df()
            except Exception:
                continue

            for col in sample.columns:
                dtype = str(sample[col].dtype)
                non_null = int(sample[col].notna().sum())
                rows.append(
                    {
                        "file_path": p.name,
                        "column": col,
                        "dtype": dtype,
                        "non_null_count": non_null,
                    }
                )

    return pd.DataFrame(rows)


def aggregate_schema(sample_limit: int = 1000) -> pd.DataFrame:
    """Aggregates column metadata across files into a compact overview.

    Returns a DataFrame indexed by column name with counts of files present, total
    non-null values (sampled), and observed dtypes as a comma-separated string.
    """
    df = scan_parquet_schema(sample_limit=sample_limit)
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby("column")
        .agg(
            files_present=("file_path", "nunique"),
            total_non_null=("non_null_count", "sum"),
            dtypes=("dtype", lambda s: ",".join(sorted(set(s))))
        )
        .reset_index()
    )
    return summary


def get_column_mapping_df(sample_limit: int = 1000) -> pd.DataFrame:
    """Convenience wrapper used by the UI to obtain aggregated schema summary."""
    return aggregate_schema(sample_limit=sample_limit)
