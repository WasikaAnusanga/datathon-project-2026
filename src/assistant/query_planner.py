import pandas as pd
from typing import Dict, Any, Optional, Tuple
from src.assistant.schema_registry import SchemaRegistry
from src.assistant.intent_router import QueryIntent, IntentCompiler
from src.assistant.llm_client import LLMClient
from src.assistant.sql_validator import SQLGlotValidator
from src.assistant.executor import DuckDBExecutor


class AssistantResult:
    """
    Standardized AI Mobility Assistant Result container.
    """

    def __init__(
        self,
        question: str,
        success: bool,
        intent: Optional[QueryIntent] = None,
        sql: Optional[str] = None,
        params: Optional[list] = None,
        data: Optional[pd.DataFrame] = None,
        meta: Optional[Dict[str, Any]] = None,
        clarification_prompt: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        self.question = question
        self.success = success
        self.intent = intent
        self.sql = sql
        self.params = params or []
        self.data = data if data is not None else pd.DataFrame()
        self.meta = meta or {}
        self.clarification_prompt = clarification_prompt
        self.error_message = error_message


class QueryPlanner:
    """
    Core Query Planner coordinating Natural Language -> QueryIntent -> Deterministic SQL -> AST Validation -> DuckDB.
    """

    def __init__(
        self,
        schema_registry: Optional[SchemaRegistry] = None,
        executor: Optional[DuckDBExecutor] = None,
        use_sample: bool = False,
    ):
        self.schema_registry = schema_registry or SchemaRegistry()
        self.executor = executor or DuckDBExecutor(use_sample=use_sample, schema_registry=self.schema_registry)
        self.llm_client = LLMClient(schema_registry=self.schema_registry)
        self.compiler = IntentCompiler(schema_registry=self.schema_registry)
        self.sql_validator = SQLGlotValidator(schema_registry=self.schema_registry)

    def process_question(self, question: str) -> AssistantResult:
        # Step 1: Parse question to structured QueryIntent
        try:
            intent = self.llm_client.parse_question_to_intent(question)
        except Exception as e:
            return AssistantResult(
                question=question,
                success=False,
                error_message=f"Failed to parse question intent: {str(e)}",
            )

        # Step 2: Handle Out-of-Scope Intent
        if intent.is_out_of_scope:
            return AssistantResult(
                question=question,
                success=False,
                intent=intent,
                error_message=intent.out_of_scope_reason,
            )

        # Step 3: Handle Ambiguous Intent / Clarification Needed
        if intent.requires_clarification:
            return AssistantResult(
                question=question,
                success=False,
                intent=intent,
                clarification_prompt=intent.clarification_prompt,
            )

        # Step 4: Deterministic SQL Compilation
        try:
            sql, params = self.compiler.compile(intent)
        except Exception as e:
            return AssistantResult(
                question=question,
                success=False,
                intent=intent,
                error_message=f"Intent compilation failed: {str(e)}",
            )

        # Step 5: SQLGlot Security & AST Validation
        val_res = self.sql_validator.validate_sql(sql)
        if not val_res.is_valid:
            return AssistantResult(
                question=question,
                success=False,
                intent=intent,
                sql=sql,
                error_message=f"SQL Security Validation failed: {val_res.error_message}",
            )

        # Step 6: DuckDB Execution
        df, exec_meta = self.executor.execute_query(sql, tuple(params) if params else None)

        if not exec_meta["success"]:
            return AssistantResult(
                question=question,
                success=False,
                intent=intent,
                sql=sql,
                params=params,
                error_message=f"DuckDB Execution Error: {exec_meta['error']}",
            )

        return AssistantResult(
            question=question,
            success=True,
            intent=intent,
            sql=sql,
            params=params,
            data=df,
            meta=exec_meta,
        )
