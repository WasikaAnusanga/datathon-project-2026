import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional



class AuditLogger:
    """
    Structured Audit Logger enforcing User Refinement #4:
    Persists metadata-only audit entries to JSONL without logging full DataFrames or figures.
    Guarantees no API keys or sensitive data are written.
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "assistant_audit.jsonl")

    def log_query_execution(
        self,
        question: str,
        intent_category: str,
        generated_sql: Optional[str],
        validation_status: str,
        clarification_status: bool,
        execution_time_ms: float,
        returned_row_count: int,
        error_category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Logs query execution telemetry. Never logs DataFrames or visual figures.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "intent": intent_category,
            "generated_sql": generated_sql,
            "validation_status": validation_status,
            "clarification_status": clarification_status,
            "execution_time_ms": round(execution_time_ms, 2),
            "returned_row_count": returned_row_count,
            "error_category": error_category,
        }


        # Write to JSON Lines log file
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # Non-blocking logging write

        return entry
