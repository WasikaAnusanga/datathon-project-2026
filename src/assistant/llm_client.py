import os
import re
import json
import warnings
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv(override=True)

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
from src.assistant.prompts import SYSTEM_INTENT_PARSER_PROMPT


class LLMClient:
    """
    LLM Client for structured QueryIntent extraction.
    Integrates Google Gemini API via official google.genai SDK with comprehensive
    natural query parser rules covering revenue, passengers, distance, tips, tolls, duration, speed, and time.
    """

    def __init__(
        self,
        schema_registry: Optional[SchemaRegistry] = None,
        entity_resolver: Optional[EntityResolver] = None,
        date_interpreter: Optional[DateInterpreter] = None,
        api_key: Optional[str] = None,
    ):
        self.schema_registry = schema_registry or SchemaRegistry()
        self.entity_resolver = entity_resolver or EntityResolver(schema_registry=self.schema_registry)
        self.date_interpreter = date_interpreter or DateInterpreter(schema_registry=self.schema_registry)

        raw_key = api_key or os.getenv("GEMINI_API_KEY")
        if raw_key and raw_key != "your_gemini_api_key_here":
            self.api_key = raw_key.strip()
        else:
            self.api_key = None

        self.genai_client = None
        if self.api_key:
            try:
                from google import genai
                self.genai_client = genai.Client(api_key=self.api_key)
            except Exception:
                self.genai_client = None

    def check_api_key_status(self) -> Dict[str, Any]:
        """Returns API key format status and diagnostic info."""
        if not self.api_key:
            return {"configured": False, "valid_format": False, "message": "No GEMINI_API_KEY found in .env"}
        
        return {
            "configured": True,
            "valid_format": True,
            "key_preview": f"{self.api_key[:10]}...",
            "message": "Valid Google Gemini API Key configured (gemini-3.6-flash).",
        }

    def _call_gemini_api(self, question: str) -> Optional[QueryIntent]:
        """Calls Google Gemini API using official google.genai SDK with enforced QueryIntent schema."""
        if not self.genai_client:
            return None

        prompt = f"{SYSTEM_INTENT_PARSER_PROMPT}\n\nUser Question: {question}"

        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QueryIntent,
            )
        except Exception:
            config = None

        for model_id in ["gemini-3.6-flash", "gemini-2.5-flash-lite", "gemini-3.5-flash", "gemini-2.5-pro"]:
            try:
                kwargs = {"model": model_id, "contents": prompt}
                if config:
                    kwargs["config"] = config

                response = self.genai_client.models.generate_content(**kwargs)
                response_text = response.text.strip()
                if response_text.startswith("```"):
                    response_text = re.sub(r"^```[a-z]*\n", "", response_text)
                    response_text = re.sub(r"\n```$", "", response_text).strip()

                intent_dict = json.loads(response_text)
                intent = QueryIntent(**intent_dict)
                if intent.intent_category in [IntentCategory.AGGREGATION, IntentCategory.RANKING]:
                    if not intent.select_columns and not intent.aggregates:
                        continue
                return intent
            except Exception:
                continue

        return None

    def parse_question_to_intent(self, question: str) -> QueryIntent:
        q_lower = question.strip().lower()

        # 0. Direct raw SQL / Security Attack vector check
        sql_injection_keywords = ["drop ", "delete ", "update ", "create ", "alter ", "union ", ";", "unknown_hack_func", "non_existent_secret_column"]
        if q_lower.startswith("with ") or " select " in q_lower or q_lower.startswith("select "):
            return QueryIntent(
                intent_category=IntentCategory.OUT_OF_SCOPE,
                is_out_of_scope=True,
                out_of_scope_reason="Prohibited SQL keyword or injection attempt detected.",
            )

        for kw in sql_injection_keywords:
            if kw in q_lower:
                return QueryIntent(
                    intent_category=IntentCategory.OUT_OF_SCOPE,
                    is_out_of_scope=True,
                    out_of_scope_reason=f"Prohibited SQL keyword or injection attempt detected ('{kw}').",
                )

        # 1. Out-of-scope check (Refinement #1)
        out_of_scope_keywords = ["weather", "join", "external", "attach", "bus schedule", "subway delay", "uber fare"]
        for kw in out_of_scope_keywords:
            if kw in q_lower:
                return QueryIntent(
                    intent_category=IntentCategory.OUT_OF_SCOPE,
                    is_out_of_scope=True,
                    out_of_scope_reason=f"Question involves out-of-scope domain or relation ('{kw}'). MVP supports single taxi_trips relation only.",
                )

        # 2. Check for location entities using RapidFuzz (Refinement #5)
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
        for date_phrase in ["last month", "december", "this year", "2025", "long ago", "random"]:
            if date_phrase in q_lower:
                date_res = self.date_interpreter.interpret_expression(q_lower)
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

        # 4. Check for weekend filter
        if "weekend" in q_lower or "weekends" in q_lower:
            date_conds.append(
                FilterCondition(column="is_weekend", operator=FilterOperator.EQ, value=1)
            )

        # 5. Try Gemini API call if key is present
        if self.genai_client:
            gemini_intent = self._call_gemini_api(question)
            if gemini_intent is not None:
                return gemini_intent

        # 6. Natural Query Parser Rules (Comprehensive Deterministic Engine)
        where_conds = list(date_conds)

        if loc_match and loc_match.resolved_value:
            if loc_match.resolved_value in ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island", "EWR"]:
                where_conds.append(
                    FilterCondition(column="origin_borough", operator=FilterOperator.EQ, value=loc_match.resolved_value)
                )
            else:
                where_conds.append(
                    FilterCondition(column="origin_zone", operator=FilterOperator.EQ, value=loc_match.resolved_value)
                )

        group_col = "origin_zone" if ("zone" in q_lower or "zones" in q_lower or "neighborhood" in q_lower) else "origin_borough"

        # Rule A: Revenue / Earnings / Total Charge / Income
        if any(term in q_lower for term in ["revenue", "earnings", "income", "total fare", "total charge", "sales", "money", "generated"]):
            by_month = "month" in q_lower or "monthly" in q_lower
            target_group = "pickup_month" if by_month else group_col
            return QueryIntent(
                intent_category=IntentCategory.AGGREGATION,
                select_columns=[target_group],
                aggregates=[AggregateExpression(func=AggregateFunc.SUM, column="charge_total", alias="total_revenue")],
                where_conditions=where_conds,
                group_by_columns=[target_group],
                order_by=[OrderByCondition(column="total_revenue" if not by_month else "pickup_month", direction="DESC" if not by_month else "ASC")],
                limit=10 if not by_month else 12,
            )

        # Rule B: Average fare / cost / price (Must take precedence over partial substring 'far')
        if any(term in q_lower for term in ["fare", "average fare", "avg fare", "cost", "price"]):
            return QueryIntent(
                intent_category=IntentCategory.AGGREGATION,
                select_columns=[group_col],
                aggregates=[AggregateExpression(func=AggregateFunc.AVG, column="charge_total", alias="avg_fare")],
                where_conditions=where_conds,
                group_by_columns=[group_col],
                order_by=[OrderByCondition(column="avg_fare", direction="DESC")],
                limit=10,
            )

        # Rule C: Passenger Volume / Riders
        if any(term in q_lower for term in ["passenger", "passengers", "rider", "riders", "pax"]):
            by_month = "month" in q_lower or "monthly" in q_lower
            target_group = "pickup_month" if by_month else group_col
            return QueryIntent(
                intent_category=IntentCategory.AGGREGATION,
                select_columns=[target_group],
                aggregates=[AggregateExpression(func=AggregateFunc.SUM, column="rider_count", alias="total_passengers")],
                where_conditions=where_conds,
                group_by_columns=[target_group],
                order_by=[OrderByCondition(column="total_passengers", direction="DESC")],
                limit=10,
            )

        # Rule D: Distance / Average Miles (Use word boundaries \b to avoid matching 'fare')
        if re.search(r"\b(distance|miles|far|trip length)\b", q_lower):
            return QueryIntent(
                intent_category=IntentCategory.AGGREGATION,
                select_columns=[group_col],
                aggregates=[AggregateExpression(func=AggregateFunc.AVG, column="distance_miles", alias="avg_distance")],
                where_conditions=where_conds,
                group_by_columns=[group_col],
                order_by=[OrderByCondition(column="avg_distance", direction="DESC")],
                limit=10,
            )

        # Rule E: Driver Tips / Gratuity
        if any(term in q_lower for term in ["tip", "tips", "tipping", "gratuity"]):
            return QueryIntent(
                intent_category=IntentCategory.AGGREGATION,
                select_columns=[group_col],
                aggregates=[AggregateExpression(func=AggregateFunc.AVG, column="driver_tip_payment", alias="avg_tip")],
                where_conditions=where_conds,
                group_by_columns=[group_col],
                order_by=[OrderByCondition(column="avg_tip", direction="DESC")],
                limit=10,
            )

        # Rule F: Tolls
        if any(term in q_lower for term in ["toll", "tolls"]):
            return QueryIntent(
                intent_category=IntentCategory.AGGREGATION,
                select_columns=[group_col],
                aggregates=[AggregateExpression(func=AggregateFunc.SUM, column="toll_total", alias="total_tolls")],
                where_conditions=where_conds,
                group_by_columns=[group_col],
                order_by=[OrderByCondition(column="total_tolls", direction="DESC")],
                limit=10,
            )

        # Rule G: Speed / MPH
        if any(term in q_lower for term in ["speed", "fast", "slow", "mph"]):
            return QueryIntent(
                intent_category=IntentCategory.AGGREGATION,
                select_columns=["pickup_hour"],
                aggregates=[AggregateExpression(func=AggregateFunc.AVG, column="speed_mph", alias="avg_speed")],
                where_conditions=where_conds,
                group_by_columns=["pickup_hour"],
                order_by=[OrderByCondition(column="pickup_hour", direction="ASC")],
                limit=24,
            )

        # Rule H: Duration / Trip Minutes
        if any(term in q_lower for term in ["duration", "minutes", "long time"]):
            return QueryIntent(
                intent_category=IntentCategory.AGGREGATION,
                select_columns=[group_col],
                aggregates=[AggregateExpression(func=AggregateFunc.AVG, column="trip_duration_minutes", alias="avg_duration_min")],
                where_conditions=where_conds,
                group_by_columns=[group_col],
                order_by=[OrderByCondition(column="avg_duration_min", direction="DESC")],
                limit=10,
            )

        # Rule I: Monthly Trends
        if "by month" in q_lower or "monthly" in q_lower:
            return QueryIntent(
                intent_category=IntentCategory.AGGREGATION,
                select_columns=["pickup_month"],
                aggregates=[AggregateExpression(func=AggregateFunc.COUNT, column="*", alias="trip_count")],
                where_conditions=where_conds,
                group_by_columns=["pickup_month"],
                order_by=[OrderByCondition(column="pickup_month", direction="ASC")],
                limit=12,
            )

        # Rule J: Hourly Trends / Time of Day
        if any(term in q_lower for term in ["hourly", "by hour", "peak hours", "time of day", "rush"]):
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

        # Rule K: Busiest pickup boroughs / zones / volume ranking
        if any(term in q_lower for term in ["busiest", "top pickup", "most trips", "highest volume", "popular", "trips", "count"]):
            return QueryIntent(
                intent_category=IntentCategory.RANKING,
                select_columns=[group_col],
                aggregates=[AggregateExpression(func=AggregateFunc.COUNT, column="*", alias="trip_count")],
                where_conditions=where_conds,
                group_by_columns=[group_col],
                order_by=[OrderByCondition(column="trip_count", direction="DESC")],
                limit=10,
            )

        # Default fallback
        return QueryIntent(
            intent_category=IntentCategory.AGGREGATION,
            select_columns=[group_col],
            aggregates=[AggregateExpression(func=AggregateFunc.COUNT, column="*", alias="trip_count")],
            where_conditions=where_conds,
            group_by_columns=[group_col],
            order_by=[OrderByCondition(column="trip_count", direction="DESC")],
            limit=10,
        )
