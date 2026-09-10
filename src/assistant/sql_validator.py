import sqlglot
import sqlglot.expressions as exp
from typing import Tuple, List, Set, Optional
from src.assistant.schema_registry import SchemaRegistry


class SQLValidatorResult:

    def __init__(
        self,
        is_valid: bool,
        error_message: Optional[str] = None,
        extracted_columns: Optional[Set[str]] = None,
        extracted_aliases: Optional[Set[str]] = None,
    ):
        self.is_valid = is_valid
        self.error_message = error_message
        self.extracted_columns = extracted_columns or set()
        self.extracted_aliases = extracted_aliases or set()


class SQLGlotValidator:
    """
    SQLGlot-based AST tree security & schema validator.
    Enforces MVP single-relation read-only scope, rejects prohibited statements/AST nodes,
    whitelists safe functions, and validates columns while respecting SELECT expression aliases.
    """

    ALLOWED_FUNCTIONS = {
        "COUNT", "SUM", "AVG", "MIN", "MAX",
        "ROUND", "COALESCE", "ABS", "DATE_TRUNC",
        "EXTRACT", "MONTH", "YEAR", "DAY", "HOUR",
        "CAST", "CASE", "IFNULL", "AND", "OR", "NOT"
    }


    PROHIBITED_NODES = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Create,
        exp.Alter,
        exp.Command,
        exp.Join,
        exp.Union,
        exp.With,  # CTEs prohibited in MVP SQL scope
    )

    def __init__(self, schema_registry: Optional[SchemaRegistry] = None):
        self.schema_registry = schema_registry or SchemaRegistry()

    def validate_sql(self, sql: str) -> SQLValidatorResult:
        sql_clean = sql.strip()

        # 1. Parse statement count check (reject multi-statement injection)
        try:
            parsed_statements = sqlglot.parse(sql_clean)
        except Exception as e:
            return SQLValidatorResult(is_valid=False, error_message=f"SQL Syntax Error: {str(e)}")

        if len(parsed_statements) == 0:
            return SQLValidatorResult(is_valid=False, error_message="Empty SQL statement.")
        if len(parsed_statements) > 1:
            return SQLValidatorResult(is_valid=False, error_message="Multi-statement SQL injection detected.")

        ast = parsed_statements[0]
        if ast is None or not isinstance(ast, exp.Select):
            return SQLValidatorResult(is_valid=False, error_message="Only single-SELECT queries are allowed.")

        # 2. Check table relation (must be single 'taxi_trips' table view)
        tables = list(ast.find_all(exp.Table))
        if len(tables) != 1:
            return SQLValidatorResult(is_valid=False, error_message=f"Queries must target exactly 1 relation ('taxi_trips'), found {len(tables)}.")

        target_table = tables[0].name
        canonical_table = self.schema_registry.get_table_name()
        if target_table.lower() != canonical_table.lower():
            return SQLValidatorResult(is_valid=False, error_message=f"Invalid target table '{target_table}'. Must target '{canonical_table}'.")

        # 3. Prohibited AST nodes check across full AST tree
        for prohibited in self.PROHIBITED_NODES:
            if list(ast.find_all(prohibited)):
                return SQLValidatorResult(
                    is_valid=False,
                    error_message=f"Prohibited SQL feature detected: {prohibited.__name__} is not allowed in MVP scope.",
                )

        # 4. Extract SELECT aliases first (Refinement #3)
        select_aliases: Set[str] = set()
        for alias in ast.find_all(exp.Alias):
            if isinstance(alias.parent, exp.Select) or any(isinstance(p, exp.Select) for p in alias.ancestors):
                alias_name = alias.alias
                if alias_name:
                    select_aliases.add(alias_name)

        # 5. Validate function calls against function whitelist
        for func in ast.find_all(exp.Func):
            func_name = func.key.upper()
            if func_name not in self.ALLOWED_FUNCTIONS:
                return SQLValidatorResult(
                    is_valid=False,
                    error_message=f"Unsafe or unwhitelisted function call '{func_name}' is prohibited.",
                )

        # 6. Validate column references (physical dataset columns OR SELECT aliases)
        extracted_columns: Set[str] = set()
        for col_node in ast.find_all(exp.Column):
            col_name = col_node.name
            if not col_name:
                continue

            extracted_columns.add(col_name)

            # Check column validity (physical column OR registered SELECT alias)
            is_valid_col = self.schema_registry.validate_column_or_alias(col_name, select_aliases)
            if not is_valid_col:
                return SQLValidatorResult(
                    is_valid=False,
                    error_message=f"Invalid or non-existent column reference '{col_name}'.",
                )

        return SQLValidatorResult(
            is_valid=True,
            error_message=None,
            extracted_columns=extracted_columns,
            extracted_aliases=select_aliases,
        )
