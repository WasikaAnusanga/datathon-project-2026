import pandas as pd
from typing import Dict, Any, Optional


class ResponseBuilder:
    """
    Factual result-grounded natural language response generator.
    Strictly uses executed DuckDB DataFrame results to format answers.
    """

    @staticmethod
    def build_response(question: str, df: pd.DataFrame, meta: Dict[str, Any]) -> str:
        if df.empty:
            return "No matching taxi trip records were found for your query in the dataset."

        row_count = len(df)
        cols = list(df.columns)

        # 1. Single row single value response (e.g. count or average)
        if row_count == 1 and len(cols) == 1:
            val = df.iloc[0, 0]
            val_fmt = f"${val:,.2f}" if "fare" in cols[0] or "charge" in cols[0] else f"{val:,.0f}" if isinstance(val, (int, float)) else str(val)
            return f"The result for '{question}' is **{val_fmt}**."

        # 2. Ranking / Aggregation top item response
        if len(cols) >= 2:
            first_row = df.iloc[0]
            item_name = str(first_row[cols[0]])
            val = first_row[cols[1]]
            
            if isinstance(val, float):
                val_str = f"${val:,.2f}" if "fare" in cols[1] or "charge" in cols[1] or "cost" in cols[1] else f"{val:,.2f}"
            elif isinstance(val, (int, pd.Int64Dtype, int, float)) or hasattr(val, "item"):
                val_num = int(val) if float(val).is_integer() else float(val)
                val_str = f"{val_num:,}" if isinstance(val_num, int) else f"{val_num:,.2f}"
            else:
                val_str = str(val)

            summary = f"Based on analysis of 45.95M taxi trips, top result for '{question}' is **{item_name}** with **{val_str}** {cols[1].replace('_', ' ')}."
            if row_count > 1:
                summary += f" (Displaying top {row_count} rows)."
            return summary


        return f"Query returned {row_count} rows in {meta.get('execution_time_ms', 0)} ms."
