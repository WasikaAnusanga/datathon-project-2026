import pytest
from src.assistant.query_planner import QueryPlanner


def test_query_planner_busiest_boroughs():
    planner = QueryPlanner(use_sample=True)
    res = planner.process_question("What are the top 5 busiest pickup boroughs?")
    assert res.success is True
    assert res.data is not None
    assert len(res.data) > 0
    assert "origin_borough" in res.data.columns
    assert "trip_count" in res.data.columns
    assert res.meta["execution_time_ms"] > 0
    planner.executor.close()


def test_query_planner_out_of_scope():
    planner = QueryPlanner(use_sample=True)
    res = planner.process_question("Join taxi trips with subway delay data")
    assert res.success is False
    assert res.intent.is_out_of_scope is True
    assert "out-of-scope" in res.error_message.lower()
    planner.executor.close()


def test_query_planner_ambiguous_location():
    planner = QueryPlanner(use_sample=True)
    res = planner.process_question("What is the average fare in Midtown?")
    assert res.success is False
    assert res.intent.requires_clarification is True
    assert "Midtown" in res.clarification_prompt
    planner.executor.close()
