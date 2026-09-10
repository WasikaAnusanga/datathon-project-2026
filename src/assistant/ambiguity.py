from typing import List, Dict, Any, Optional, Tuple
from rapidfuzz import process, fuzz
from src.assistant.schema_registry import SchemaRegistry


class MatchStatus(str):
    EXACT = "EXACT"
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"


class ResolutionResult:

    def __init__(
        self,
        status: str,
        resolved_value: Optional[str] = None,
        confidence_score: float = 0.0,
        candidates: Optional[List[str]] = None,
        interpretation: Optional[str] = None,
        clarification_prompt: Optional[str] = None,
    ):
        self.status = status
        self.resolved_value = resolved_value
        self.confidence_score = confidence_score
        self.candidates = candidates or []
        self.interpretation = interpretation
        self.clarification_prompt = clarification_prompt


class EntityResolver:
    """
    RapidFuzz Entity Resolver enforcing User Refinement #5 rules:
    - Exact unique match -> automatically resolve
    - High-confidence match (score >= 85, margin >= 10) -> resolve + interpretation
    - Multiple close matches (margin < 10) -> ask clarification
    - Low-confidence match (score < 70) -> unknown entity response
    """

    def __init__(self, vocabulary: Optional[List[str]] = None, schema_registry: Optional[SchemaRegistry] = None):
        self.schema_registry = schema_registry or SchemaRegistry()
        self.vocabulary = vocabulary or self._load_default_vocabulary()

    def _load_default_vocabulary(self) -> List[str]:
        # TLC Boroughs + key iconic zones
        boroughs = ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island", "EWR"]
        key_zones = [
            "JFK Airport",
            "LaGuardia Airport",
            "Times Sq/Theatre District",
            "Midtown Center",
            "Midtown East",
            "Upper East Side South",
            "Upper West Side South",
            "Financial District South",
            "East Village",
            "Greenwich Village South",
            "DUMBO/Vinegar Hill",
            "Williamsburg (North Side)",
            "Astoria",
            "Flushing",
        ]
        return list(set(boroughs + key_zones))

    def resolve_location(self, query_term: str) -> ResolutionResult:
        query_clean = query_term.strip()
        if not query_clean:
            return ResolutionResult(status=MatchStatus.UNKNOWN, clarification_prompt="Empty location query.")

        # 1. Exact match check (case-insensitive)
        for cand in self.vocabulary:
            if cand.lower() == query_clean.lower():
                return ResolutionResult(
                    status=MatchStatus.EXACT,
                    resolved_value=cand,
                    confidence_score=100.0,
                    interpretation=f"Exact match: '{cand}'",
                )

        # 2. RapidFuzz extraction
        matches = process.extract(
            query_clean, self.vocabulary, scorer=fuzz.WRatio, limit=5
        )

        if not matches:
            return ResolutionResult(
                status=MatchStatus.UNKNOWN,
                clarification_prompt=f"Unknown location '{query_term}'. Please specify a valid NYC Borough or Taxi Zone.",
            )

        top_match, top_score, _ = matches[0]

        # Low-confidence threshold (< 70)
        if top_score < 70:
            return ResolutionResult(
                status=MatchStatus.UNKNOWN,
                confidence_score=top_score,
                clarification_prompt=f"Location '{query_term}' could not be identified with confidence. Did you mean one of: {', '.join([m[0] for m in matches[:3]])}?",
            )

        # Check margin over runner-up
        if len(matches) > 1:
            second_match, second_score, _ = matches[1]
            margin = top_score - second_score
        else:
            margin = 100.0

        # High-confidence with strong margin (score >= 85, margin >= 10)
        if top_score >= 85 and margin >= 10:
            return ResolutionResult(
                status=MatchStatus.HIGH_CONFIDENCE,
                resolved_value=top_match,
                confidence_score=top_score,
                interpretation=f"Interpreted '{query_term}' as '{top_match}' (confidence: {top_score:.1f}%)",
            )

        # Ambiguous / multiple close matches (margin < 10)
        close_candidates = [m[0] for m in matches if (top_score - m[1]) < 10]
        prompt = f"Multiple location matches found for '{query_term}': {', '.join(close_candidates)}. Which specific location did you mean?"
        
        return ResolutionResult(
            status=MatchStatus.AMBIGUOUS,
            confidence_score=top_score,
            candidates=close_candidates,
            clarification_prompt=prompt,
        )


class DateInterpreter:
    """
    Dataset-time interpreter enforcing User Refinement #6:
    Anchors relative dates ('last month', 'this year', 'December')
    against dataset range [2025-04-01 to 2026-03-31].
    """

    def __init__(self, schema_registry: Optional[SchemaRegistry] = None):
        self.schema_registry = schema_registry or SchemaRegistry()
        self.bounds = self.schema_registry.get_dataset_temporal_bounds()

    def interpret_expression(self, expr: str) -> Dict[str, Any]:
        expr_clean = expr.strip().lower()

        # Last month -> 2026-03
        if "last month" in expr_clean or "previous month" in expr_clean:
            return {
                "where_column": "pickup_month",
                "operator": "=",
                "value": 3,
                "year": 2026,
                "interpretation": "Interpreted 'last month' as March 2026 (dataset bounds: April 2025 - March 2026)",
                "requires_clarification": False,
            }

        # December -> 2025-12
        if "december" in expr_clean or "dec" in expr_clean:
            return {
                "where_column": "pickup_month",
                "operator": "=",
                "value": 12,
                "year": 2025,
                "interpretation": "Interpreted 'December' as December 2025",
                "requires_clarification": False,
            }

        # This year -> 2026
        if "this year" in expr_clean or "current year" in expr_clean:
            return {
                "where_column": "pickup_date",
                "operator": "BETWEEN",
                "value": ("2026-01-01", "2026-03-31"),
                "interpretation": "Interpreted 'this year' as 2026 (Jan 1, 2026 - Mar 31, 2026)",
                "requires_clarification": False,
            }

        # Last year -> 2025
        if "last year" in expr_clean or "2025" in expr_clean:
            return {
                "where_column": "pickup_date",
                "operator": "BETWEEN",
                "value": ("2025-04-01", "2025-12-31"),
                "interpretation": "Interpreted '2025' as April 1, 2025 - December 31, 2025",
                "requires_clarification": False,
            }

        # Ambiguous date expression
        return {
            "where_column": None,
            "operator": None,
            "value": None,
            "interpretation": None,
            "requires_clarification": True,
            "clarification_prompt": f"Unclear temporal expression '{expr}'. Available dataset range is April 2025 to March 2026. Please specify month/year.",
        }
