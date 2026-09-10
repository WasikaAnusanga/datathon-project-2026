from enum import Enum
from typing import List, Optional, Any, Dict, Set, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator
from src.assistant.schema_registry import SchemaRegistry


class IntentCategory(str, Enum):
    AGGREGATION = "aggregation"
    RANKING = "ranking"
    FILTER_BROWSE = "filter_browse"
    ANOMALY_CHECK = "anomaly_check"
    COMPARISON = "comparison"
    OUT_OF_SCOPE = "out_of_scope"
    AMBIGUOUS = "ambiguous"


class AggregateFunc(str, Enum):
    COUNT = "COUNT"
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"


class FilterOperator(str, Enum):
    EQ = "="
    NEQ = "!="
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    IN = "IN"
    LIKE = "LIKE"
    BETWEEN = "BETWEEN"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"


class FilterCondition(BaseModel):
    column: str
    operator: FilterOperator
    value: Any  # scalar, list for IN, or tuple for BETWEEN


class AggregateExpression(BaseModel):
    func: AggregateFunc
    column: str = "*"
    alias: Optional[str] = None


class OrderByCondition(BaseModel):
    column: str
    direction: str = "DESC"  # ASC or DESC


class QueryIntent(BaseModel):
    """
    Structured, Pydantic-validated representation of user analytical intent.
    Translates deterministically into bound SQL queries within MVP scope.
    """
    intent_category: IntentCategory = IntentCategory.AGGREGATION
    select_columns: List[str] = Field(default_factory=list)
    aggregates: List[AggregateExpression] = Field(default_factory=list)
    where_conditions: List[FilterCondition] = Field(default_factory=list)
    group_by_columns: List[str] = Field(default_factory=list)
    having_conditions: List[FilterCondition] = Field(default_factory=list)
    order_by: List[OrderByCondition] = Field(default_factory=list)
    limit: int = Field(default=100, ge=1, le=1000)
    
    is_out_of_scope: bool = False
    out_of_scope_reason: Optional[str] = None
    requires_clarification: bool = False
    clarification_prompt: Optional[str] = None

    def get_query_aliases(self) -> Set[str]:
        """Returns set of aliases defined in aggregates or select columns."""
        aliases = set()
        for agg in self.aggregates:
            if agg.alias:
                aliases.add(agg.alias)
        return aliases


class IntentCompiler:
    """
    Deterministic SQL Compiler that transforms a validated QueryIntent
    into a safe, parameter-bound DuckDB SQL string within MVP scope.
    """

    def __init__(self, schema_registry: Optional[SchemaRegistry] = None):
        self.schema_registry = schema_registry or SchemaRegistry()

    def compile(self, intent: QueryIntent) -> Tuple[str, List[Any]]:
        """
        Compiles a QueryIntent into (sql_string, params_list).
        No string concatenation of raw user input into SQL values.
        """
        if intent.is_out_of_scope:
            raise ValueError(f"Cannot compile out-of-scope query intent: {intent.out_of_scope_reason}")
        if intent.requires_clarification:
            raise ValueError(f"Cannot compile ambiguous query intent: {intent.clarification_prompt}")

        table_name = self.schema_registry.get_table_name()
        query_aliases = intent.get_query_aliases()

        # 1. SELECT clause
        select_parts = []
        
        # Add plain select columns
        for col in intent.select_columns:
            canonical = self.schema_registry.resolve_column(col) or col
            select_parts.append(canonical)
            
        # Add aggregates
        for agg in intent.aggregates:
            if agg.column == "*":
                agg_target = "*"
            else:
                agg_target = self.schema_registry.resolve_column(agg.column) or agg.column
            
            expr = f"{agg.func.value}({agg_target})"
            if agg.alias:
                expr += f" AS {agg.alias}"
            select_parts.append(expr)

        if not select_parts:
            select_parts = ["*"]

        select_str = ", ".join(select_parts)
        sql = f"SELECT {select_str}\nFROM {table_name}"
        params: List[Any] = []

        # 2. WHERE clause
        if intent.where_conditions:
            where_parts = []
            for cond in intent.where_conditions:
                col_canonical = self.schema_registry.resolve_column(cond.column) or cond.column
                
                if cond.operator in (FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL):
                    where_parts.append(f"{col_canonical} {cond.operator.value}")
                elif cond.operator == FilterOperator.IN:
                    if isinstance(cond.value, (list, tuple)) and cond.value:
                        placeholders = ", ".join(["?"] * len(cond.value))
                        where_parts.append(f"{col_canonical} IN ({placeholders})")
                        params.extend(cond.value)
                    else:
                        where_parts.append(f"{col_canonical} = ?")
                        params.append(cond.value)
                elif cond.operator == FilterOperator.BETWEEN:
                    where_parts.append(f"{col_canonical} BETWEEN ? AND ?")
                    params.extend(cond.value)
                else:
                    where_parts.append(f"{col_canonical} {cond.operator.value} ?")
                    params.append(cond.value)

            sql += "\nWHERE " + " AND ".join(where_parts)

        # 3. GROUP BY clause
        if intent.group_by_columns:
            group_parts = []
            for col in intent.group_by_columns:
                canonical = self.schema_registry.resolve_column(col) or col
                group_parts.append(canonical)
            sql += "\nGROUP BY " + ", ".join(group_parts)

        # 4. HAVING clause
        if intent.having_conditions:
            having_parts = []
            for cond in intent.having_conditions:
                # Column in HAVING could be an aggregate or alias
                col_ref = cond.column
                if self.schema_registry.is_physical_column(col_ref):
                    col_ref = self.schema_registry.resolve_column(col_ref)
                
                having_parts.append(f"{col_ref} {cond.operator.value} ?")
                params.append(cond.value)

            sql += "\nHAVING " + " AND ".join(having_parts)

        # 5. ORDER BY clause
        if intent.order_by:
            order_parts = []
            for ob in intent.order_by:
                col_ref = ob.column
                if self.schema_registry.is_physical_column(col_ref):
                    col_ref = self.schema_registry.resolve_column(col_ref)
                direction = "DESC" if ob.direction.upper() == "DESC" else "ASC"
                order_parts.append(f"{col_ref} {direction}")
            sql += "\nORDER BY " + ", ".join(order_parts)

        # 6. LIMIT clause
        sql += f"\nLIMIT {intent.limit}"

        return sql, params
