import pytest
import os
from src.assistant.schema_registry import SchemaRegistry


def test_schema_registry_loading():
    registry = SchemaRegistry()
    assert len(registry.physical_columns) == 60
    assert registry.get_table_name() == "taxi_trips"


def test_physical_column_lookup():
    registry = SchemaRegistry()
    assert registry.is_physical_column("origin_zone") is True
    assert registry.is_physical_column("ORIGIN_ZONE") is True
    assert registry.is_physical_column("non_existent_col") is False


def test_synonym_resolution():
    registry = SchemaRegistry()
    assert registry.resolve_column("passenger_count") == "rider_count"
    assert registry.resolve_column("trip_distance") == "distance_miles"
    assert registry.resolve_column("pu_zone") == "origin_zone"
    assert registry.resolve_column("total_amount") == "charge_total"


def test_alias_validation():
    registry = SchemaRegistry()
    # Physical column is valid
    assert registry.validate_column_or_alias("origin_zone", set()) is True
    
    # Query alias (e.g. trip_count in SELECT origin_zone, COUNT(*) AS trip_count ...) is valid
    query_aliases = {"trip_count", "avg_fare"}
    assert registry.validate_column_or_alias("trip_count", query_aliases) is True
    assert registry.validate_column_or_alias("TRIP_COUNT", query_aliases) is True
    
    # Unknown column/alias is rejected
    assert registry.validate_column_or_alias("malicious_column", query_aliases) is False


def test_dataset_temporal_bounds():
    registry = SchemaRegistry()
    bounds = registry.get_dataset_temporal_bounds()
    assert bounds["dataset_min_date"] == "2025-04-01"
    assert bounds["dataset_max_date"] == "2026-03-31"
    assert bounds["default_anchor_date"] == "2026-03-31"
    assert 2025 in bounds["available_years"]
