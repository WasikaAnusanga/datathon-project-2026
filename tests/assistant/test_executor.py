import pytest
from src.assistant.executor import DuckDBExecutor


def test_executor_sample_query():
    executor = DuckDBExecutor(use_sample=True)
    df, meta = executor.execute_query("SELECT origin_borough, COUNT(*) AS trip_count FROM taxi_trips GROUP BY origin_borough ORDER BY trip_count DESC")
    
    assert meta["success"] is True
    assert meta["returned_row_count"] > 0
    assert "origin_borough" in df.columns
    assert "trip_count" in df.columns
    assert meta["execution_time_ms"] > 0
    executor.close()


def test_executor_hard_limit():
    executor = DuckDBExecutor(use_sample=True)
    df, meta = executor.execute_query("SELECT * FROM taxi_trips")
    assert meta["success"] is True
    assert meta["returned_row_count"] <= 1000
    executor.close()


def test_executor_invalid_sql():
    executor = DuckDBExecutor(use_sample=True)
    df, meta = executor.execute_query("SELECT non_existent_col FROM taxi_trips")
    assert meta["success"] is False
    assert meta["error"] is not None
    assert len(df) == 0
    executor.close()
