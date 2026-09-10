import pytest
from src.assistant.intent_router import (
    QueryIntent,
    IntentCompiler,
    IntentCategory,
    AggregateExpression,
    AggregateFunc,
    FilterCondition,
    FilterOperator,
    OrderByCondition,
)
from src.assistant.executor import DuckDBExecutor


def test_intent_compilation_simple_aggregation():
    intent = QueryIntent(
        intent_category=IntentCategory.AGGREGATION,
        select_columns=["origin_borough"],
        aggregates=[AggregateExpression(func=AggregateFunc.COUNT, column="*", alias="trip_count")],
        where_conditions=[
            FilterCondition(column="pickup_hour", operator=FilterOperator.EQ, value=8)
        ],
        group_by_columns=["origin_borough"],
        order_by=[OrderByCondition(column="trip_count", direction="DESC")],
        limit=10,
    )

    compiler = IntentCompiler()
    sql, params = compiler.compile(intent)

    assert "SELECT origin_borough, COUNT(*) AS trip_count" in sql
    assert "FROM taxi_trips" in sql
    assert "WHERE pickup_hour = ?" in sql
    assert "GROUP BY origin_borough" in sql
    assert "ORDER BY trip_count DESC" in sql
    assert "LIMIT 10" in sql
    assert params == [8]


def test_intent_execution_with_executor():
    intent = QueryIntent(
        intent_category=IntentCategory.RANKING,
        select_columns=["origin_borough"],
        aggregates=[AggregateExpression(func=AggregateFunc.AVG, column="charge_total", alias="avg_fare")],
        group_by_columns=["origin_borough"],
        order_by=[OrderByCondition(column="avg_fare", direction="DESC")],
        limit=5,
    )

    compiler = IntentCompiler()
    sql, params = compiler.compile(intent)

    executor = DuckDBExecutor(use_sample=True)
    df, meta = executor.execute_query(sql, tuple(params))

    assert meta["success"] is True
    assert len(df) > 0
    assert "origin_borough" in df.columns
    assert "avg_fare" in df.columns
    executor.close()


def test_out_of_scope_intent():
    intent = QueryIntent(
        intent_category=IntentCategory.OUT_OF_SCOPE,
        is_out_of_scope=True,
        out_of_scope_reason="External weather data relation is not supported in MVP scope.",
    )
    compiler = IntentCompiler()
    with pytest.raises(ValueError, match="out-of-scope"):
        compiler.compile(intent)


def test_ambiguous_intent():
    intent = QueryIntent(
        intent_category=IntentCategory.AMBIGUOUS,
        requires_clarification=True,
        clarification_prompt="Multiple zones match 'Airport': JFK Airport, LaGuardia Airport. Which one did you mean?",
    )
    compiler = IntentCompiler()
    with pytest.raises(ValueError, match="ambiguous"):
        compiler.compile(intent)
