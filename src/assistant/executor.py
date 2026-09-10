import duckdb
import time
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from src.assistant.schema_registry import SchemaRegistry


class DuckDBExecutor:
    """
    High-performance DuckDB query execution engine.
    Registers read-only taxi_trips view over enriched Parquet files.
    Enforces row limits and returns structured execution metrics.
    """

    DEFAULT_PREVIEW_ROWS = 100
    DEFAULT_QUERY_LIMIT = 1000
    HARD_MAX_RESULT_ROWS = 1000

    def __init__(self, use_sample: bool = False, schema_registry: Optional[SchemaRegistry] = None):
        self.schema_registry = schema_registry or SchemaRegistry()
        self.use_sample = use_sample
        self.con = duckdb.connect(database=":memory:", read_only=False)
        self._register_view()

    def _register_view(self) -> None:
        """Register taxi_trips view in DuckDB pointing to parquet data."""
        if self.use_sample:
            parquet_path = self.schema_registry.get_sample_fixture_path()
        else:
            parquet_path = self.schema_registry.get_data_path()

        table_name = self.schema_registry.get_table_name()

        # Create view over parquet file
        view_sql = f"CREATE VIEW {table_name} AS SELECT * FROM read_parquet('{parquet_path}')"
        self.con.execute(view_sql)

    def execute_query(self, sql: str, params: Optional[Tuple[Any, ...]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes a SQL query against DuckDB and returns (df, meta).
        Enforces HARD_MAX_RESULT_ROWS limit safely.
        """
        start_time = time.perf_counter()

        try:
            if params:
                rel = self.con.execute(sql, params)
            else:
                rel = self.con.execute(sql)

            # Fetch up to HARD_MAX_RESULT_ROWS + 1 to check for truncation
            df = rel.fetch_df_chunk(self.HARD_MAX_RESULT_ROWS) if hasattr(rel, "fetch_df_chunk") else rel.fetchdf()

            if len(df) > self.HARD_MAX_RESULT_ROWS:
                df = df.iloc[: self.HARD_MAX_RESULT_ROWS].copy()
                truncated = True
            else:
                truncated = False

            exec_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            meta = {
                "execution_time_ms": exec_time_ms,
                "returned_row_count": len(df),
                "truncated": truncated,
                "success": True,
                "error": None,
            }
            return df, meta

        except Exception as e:
            exec_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            meta = {
                "execution_time_ms": exec_time_ms,
                "returned_row_count": 0,
                "truncated": False,
                "success": False,
                "error": str(e),
            }
            return pd.DataFrame(), meta

    def close(self) -> None:
        """Close connection."""
        try:
            self.con.close()
        except Exception:
            pass

