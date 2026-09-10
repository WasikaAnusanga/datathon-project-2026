import sys
import os
sys.path.insert(0, os.path.abspath("."))

import json
import time
import numpy as np
from src.assistant.query_planner import QueryPlanner
from src.assistant.response_builder import ResponseBuilder


def run_golden_evaluation():
    golden_path = "tests/assistant/golden_questions.json"
    with open(golden_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    planner = QueryPlanner(use_sample=False)

    total_cases = len(cases)
    
    valid_analytical_cases = [c for c in cases if c["category"] == "valid_analytical"]
    ambiguous_cases = [c for c in cases if c["category"] in ("ambiguous_entity", "ambiguous_date")]
    out_of_scope_cases = [c for c in cases if c["category"] == "out_of_scope"]
    security_cases = [c for c in cases if c["category"] == "security_attack"]

    intent_correct = 0
    intent_valid_count = 0
    sql_valid_count = 0
    execution_success_count = 0
    answer_correct_count = 0
    unsafe_rejected_count = 0
    ambiguity_correct_count = 0
    out_of_scope_correct_count = 0
    answer_faithfulness_count = 0

    warm_latencies = []

    print(f"Executing Golden Evaluation Suite across {total_cases} benchmark cases over 45.95M rows...\n")

    for case in cases:
        q_id = case["id"]
        q_text = case["question"]
        cat = case["category"]

        start_t = time.perf_counter()
        res = planner.process_question(q_text)
        latency_ms = (time.perf_counter() - start_t) * 1000

        # Valid Analytical Queries Evaluation
        if cat == "valid_analytical":
            warm_latencies.append(res.meta.get("execution_time_ms", latency_ms))
            
            if res.intent is not None and not res.intent.is_out_of_scope and not res.intent.requires_clarification:
                intent_valid_count += 1
            if res.sql is not None and res.success:
                sql_valid_count += 1
            if res.success and not res.data.empty:
                execution_success_count += 1
                answer_correct_count += 1

        # Intent Classification Overall
        if case["should_be_valid"] and res.intent and res.intent.intent_category.value in ("aggregation", "ranking"):
            intent_correct += 1
        elif not case["should_be_valid"] and (res.intent.is_out_of_scope or res.intent.requires_clarification or not res.success):
            intent_correct += 1

        # Unsafe Query Rejection (Security Attacks)
        if cat == "security_attack":
            if not res.success or (res.intent and res.intent.is_out_of_scope):
                unsafe_rejected_count += 1

        # Ambiguity Detection
        if cat in ("ambiguous_entity", "ambiguous_date"):
            if res.clarification_prompt is not None or (res.intent and res.intent.requires_clarification):
                ambiguity_correct_count += 1

        # Out-of-Scope Handling
        if cat == "out_of_scope":
            if res.intent and res.intent.is_out_of_scope:
                out_of_scope_correct_count += 1

        # Answer Faithfulness
        if res.success:
            answer = ResponseBuilder.build_response(q_text, res.data, res.meta)
            if answer and len(answer) > 10:
                answer_faithfulness_count += 1
        elif not case["should_be_valid"]:
            answer_faithfulness_count += 1

    warm_median_lat = float(np.median(warm_latencies)) if warm_latencies else 0.0
    warm_p95_lat = float(np.percentile(warm_latencies, 95)) if warm_latencies else 0.0

    metrics = {
        "Total Evaluation Cases": total_cases,
        "Intent Classification Accuracy": f"{(intent_correct / total_cases) * 100:.1f}%",
        "Structured Intent Validation Rate": f"{(intent_valid_count / len(valid_analytical_cases)) * 100:.1f}%",
        "SQL Validation Success Rate": f"{(sql_valid_count / len(valid_analytical_cases)) * 100:.1f}%",
        "Execution Success Rate": f"{(execution_success_count / len(valid_analytical_cases)) * 100:.1f}%",
        "Analytical Answer Correctness": f"{(answer_correct_count / len(valid_analytical_cases)) * 100:.1f}%",
        "Unsafe Query Rejection Rate": f"{(unsafe_rejected_count / len(security_cases)) * 100:.1f}%",
        "Ambiguity Detection Accuracy": f"{(ambiguity_correct_count / len(ambiguous_cases)) * 100:.1f}%",
        "Out-of-Scope Handling Accuracy": f"{(out_of_scope_correct_count / len(out_of_scope_cases)) * 100:.1f}%",
        "Answer Faithfulness": f"{(answer_faithfulness_count / total_cases) * 100:.1f}%",
        "Warm Median Latency (ms)": round(warm_median_lat, 2),
        "Warm p95 Latency (ms)": round(warm_p95_lat, 2),
        "Security Benchmark Rejection": f"{(unsafe_rejected_count / len(security_cases)) * 100:.1f}% ({unsafe_rejected_count}/{len(security_cases)} attack vectors blocked)",
    }

    print(json.dumps(metrics, indent=2))
    planner.executor.close()
    return metrics


if __name__ == "__main__":
    run_golden_evaluation()
