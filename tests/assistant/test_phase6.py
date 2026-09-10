import os
import json
import pandas as pd
import pytest
from src.assistant.response_builder import ResponseBuilder
from src.assistant.chart_builder import ChartBuilder
from src.assistant.audit_logger import AuditLogger


def test_response_builder_summary():
    df = pd.DataFrame({
        "origin_borough": ["Manhattan", "Queens"],
        "trip_count": [250000, 180000]
    })
    resp = ResponseBuilder.build_response("What are busiest boroughs?", df, {"execution_time_ms": 45.2})
    assert "Manhattan" in resp
    assert "250,000" in resp


def test_chart_builder_figure():
    df = pd.DataFrame({
        "origin_borough": ["Manhattan", "Queens", "Brooklyn"],
        "avg_fare": [45.5, 38.2, 32.1]
    })
    fig = ChartBuilder.generate_chart(df, "Average Fare by Borough")
    assert fig is not None
    assert fig.layout.title.text is not None


def test_audit_logger_metadata_only(tmp_path):
    log_dir = str(tmp_path)
    logger = AuditLogger(log_dir=log_dir)
    
    entry = logger.log_query_execution(
        question="What is top borough?",
        intent_category="ranking",
        generated_sql="SELECT origin_borough FROM taxi_trips",
        validation_status="VALID",
        clarification_status=False,
        execution_time_ms=120.5,
        returned_row_count=5,
        error_category=None
    )
    
    assert entry["returned_row_count"] == 5
    assert "dataframe" not in entry
    assert "data" not in entry
    assert "figure" not in entry
    
    # Verify file content
    log_file = os.path.join(log_dir, "assistant_audit.jsonl")
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        line = f.readline()
        data = json.loads(line)
        assert data["question"] == "What is top borough?"
        assert data["intent"] == "ranking"
