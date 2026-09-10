import os
import re
from typing import Dict, Any, Optional
from src.assistant.schema_registry import SchemaRegistry
from src.assistant.intent_router import (

    QueryIntent,
    IntentCategory,
    AggregateExpression,
    AggregateFunc,
    FilterCondition,
    FilterOperator,
    OrderByCondition,
)
from src.assistant.ambiguity import EntityResolver, DateInterpreter, MatchStatus


class LLMClient:
    """
    LLM Client for structured QueryIntent extraction.
    Provides rule-grounded deterministic intent extraction for standard mobility questions,
    with fallbacks for out-of-scope and ambiguous queries.
    """

    def __init__(
        self,
        schema_registry: Optional[SchemaRegistry] = None,
        entity_resolver: Optional[EntityResolver] = None,
        date_interpreter: Optional[DateInterpreter] = None,
    ):
        self.schema_registry = schema_registry or SchemaRegistry()
        self.entity_resolver = entity_resolver or EntityResolver(schema_registry=self.schema_registry)
        self.date_interpreter = date_interpreter or DateInterpreter(schema_registry=self.schema_registry)


    def parse_question_to_intent(self, question: str) -> QueryIntent:
        q_lower = question.strip().lower()

        # 1. Out-of-scope check (Refinement #1)
        out_of_scope_keywords = ["weather", "join", "external", "attach", "union", "bus schedule", "subway delay", "uber fare"]
        for kw in out_of_scope_keywords:
            if kw in q_lower:
                return QueryIntent(
                    intent_category=IntentCategory.OUT_OF_SCOPE,
                    is_out_of_scope=True,
                    out_of_scope_reason=f"Question involves out-of-scope domain or relation ('{kw}'). MVP supports single taxi_trips relation only.",
                )

        # 2. Check for location entities using RapidFuzz (Refinement #5)
        # Check if question mentions a location term like "jfk", "midtown", "manhattan"
        loc_match = None
        for word in ["jfk", "laguardia", "manhattan", "queens", "brooklyn", "bronx", "staten island", "midtown"]:
            if word in q_lower:
                loc_match = self.entity_resolver.resolve_location(word)
                break

        if loc_match and loc_match.status == MatchStatus.AMBIGUOUS:
            return QueryIntent(
                intent_category=IntentCategory.AMBIGUOUS,
                requires_clarification=True,
                clarification_prompt=loc_match.clarification_prompt,
            )

        if loc_match and loc_match.status == MatchStatus.UNKNOWN:
            return QueryIntent(
                intent_category=IntentCategory.AMBIGUOUS,
                requires_clarification=True,
                clarification_prompt=loc_match.clarification_prompt,
            )

        # 3. Check for temporal relative date expressions (Refinement #6)
        date_conds = []
        for date_phrase in ["last month", "december", "this year", "2025"]:
            if date_phrase in q_lower:
                date_res = self.date_interpreter.interpret_expression(date_phrase)
                if date_res.get("requires_clarification"):
                    return QueryIntent(
                        intent_category=IntentCategory.AMBIGUOUS,
                        requires_clarification=True,
                        clarification_prompt=date_res["clarification_prompt"],
                    )
                if date_res.get("where_column"):
                    date_conds.append(
                        FilterCondition(
                            column=date_res["where_column"],
                            operator=FilterOperator(date_res["operator"]),
                            value=date_res["value"],
                        )
                    )

        # 4. Question Pattern Matching -> Deterministic QueryIntent
        where_conds = list(date_conds)

        # Add location filter if resolved
        if loc_match and loc_match.resolved_value:
            if loc_match.resolved_value in ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island", "EWR"]:
                where_conds.append(
                    FilterCondition(column="origin_borough", operator=FilterOperator.EQ, value=loc_match.resolved_value)
                )
            else:
                where_conds.append(
                    FilterCondition(column="origin_zone", operator=FilterOperator.EQ, value=loc_match.resolved_value)
                )

        # Pattern A: Busiest pickup boroughs / zones / ranking
        if any(term in q_lower for term in ["busiest", "top pickup", "most trips", "highest volume"]):
            group_col = "origin_borough" if "borough" in q_lower else "origin_zone"
            return QueryIntent(
                intent_category=IntentCategory.RANKING,
                select_columns=[group_col],
                aggregates=[AggregateExpression(func=AggregateFunc.COUNT, column="*", alias="trip_count")],
                where_conditions=where_conds,
                group_by_columns=[group_col],
                order_by=[OrderByCondition(column="trip_count", direction="DESC")],
                limit=10,
            )

        # Pattern B: Average fare / cost
        if any(term in q_lower for term in ["average fare", "avg fare", "average cost", "mean fare"]):
            group_col = "origin_borough"
            return QueryIntent(
                intent_category=IntentCategory.AGGREGATION,
                select_columns=[group_col],
                aggregates=[AggregateExpression(func=AggregateFunc.AVG, column="charge_total", alias="avg_fare")],
                where_conditions=where_conds,
                group_by_columns=[group_col],
                order_by=[OrderByCondition(column="avg_fare", direction="DESC")],
                limit=10,
            )

        # Pattern C: Hourly distribution / rush hours
        if any(term in q_lower for term in ["hourly", "by hour", "peak hours", "time of day"]):
            return QueryIntent(
                intent_category=IntentCategory.AGGREGATION,
                select_columns=["pickup_hour"],
                aggregates=[
                    AggregateExpression(func=AggregateFunc.COUNT, column="*", alias="trip_count"),
                    AggregateExpression(func=AggregateFunc.AVG, column="speed_mph", alias="avg_speed"),
                ],
                where_conditions=where_conds,
                group_by_columns=["pickup_hour"],
                order_by=[OrderByCondition(column="pickup_hour", direction="ASC")],
                limit=24,
            )

        # Default fallback intent for general aggregation questions
        return QueryIntent(
            intent_category=IntentCategory.AGGREGATION,
            select_columns=["origin_borough"],
            aggregates=[AggregateExpression(func=AggregateFunc.COUNT, column="*", alias="trip_count")],
            where_conditions=where_conds,
            group_by_columns=["origin_borough"],
            order_by=[OrderByCondition(column="trip_count", direction="DESC")],
            limit=100,
        )
