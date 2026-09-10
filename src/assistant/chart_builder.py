import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional


class ChartBuilder:
    """
    Automatic Plotly visualizer for AI Mobility Assistant query results.
    """

    @staticmethod
    def generate_chart(df: pd.DataFrame, question: str) -> Optional[go.Figure]:
        if df.empty or len(df.columns) < 2:
            return None

        cols = list(df.columns)
        x_col = cols[0]
        y_col = cols[1]

        # Determine chart type based on column types and names
        if "hour" in x_col.lower() or "date" in x_col.lower() or "month" in x_col.lower():
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                title=f"Trend: {y_col} by {x_col}",
                markers=True,
                template="plotly_dark",
            )
        elif pd.api.types.is_numeric_dtype(df[y_col]):
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                title=f"Comparison: {y_col} by {x_col}",
                color=y_col if len(df) <= 20 else None,
                template="plotly_dark",
            )
        else:
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                title=f"{question}",
                template="plotly_dark",
            )

        fig.update_layout(
            margin=dict(l=40, r=40, t=50, b=40),
            hovermode="x unified" if "hour" in x_col.lower() else "closest",
        )
        return fig
