import pytest
from src.assistant.ambiguity import EntityResolver, DateInterpreter, MatchStatus


def test_entity_resolver_exact_match():
    resolver = EntityResolver()
    res = resolver.resolve_location("Manhattan")
    assert res.status == MatchStatus.EXACT
    assert res.resolved_value == "Manhattan"
    assert res.confidence_score == 100.0


def test_entity_resolver_high_confidence_match():
    resolver = EntityResolver()
    res = resolver.resolve_location("JFK")
    assert res.status in (MatchStatus.EXACT, MatchStatus.HIGH_CONFIDENCE)
    assert res.resolved_value == "JFK Airport"
    assert res.confidence_score >= 85.0


def test_entity_resolver_ambiguous_matches():
    resolver = EntityResolver(vocabulary=["Midtown Center", "Midtown East", "Midtown West"])
    res = resolver.resolve_location("Midtown")
    assert res.status == MatchStatus.AMBIGUOUS
    assert len(res.candidates) > 1
    assert "Midtown Center" in res.candidates
    assert "Midtown East" in res.candidates
    assert "Which specific location" in res.clarification_prompt


def test_entity_resolver_unknown_location():
    resolver = EntityResolver()
    res = resolver.resolve_location("Atlantis Ocean Park")
    assert res.status == MatchStatus.UNKNOWN
    assert res.resolved_value is None
    assert res.clarification_prompt is not None


def test_date_interpreter_last_month():
    interpreter = DateInterpreter()
    res = interpreter.interpret_expression("last month")
    assert res["requires_clarification"] is False
    assert res["where_column"] == "pickup_month"
    assert res["value"] == 3
    assert res["year"] == 2026


def test_date_interpreter_december():
    interpreter = DateInterpreter()
    res = interpreter.interpret_expression("December")
    assert res["requires_clarification"] is False
    assert res["value"] == 12
    assert res["year"] == 2025


def test_date_interpreter_ambiguous_date():
    interpreter = DateInterpreter()
    res = interpreter.interpret_expression("some random Tuesday long ago")
    assert res["requires_clarification"] is True
    assert "Unclear temporal expression" in res["clarification_prompt"]
