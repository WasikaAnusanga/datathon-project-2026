import pytest
from src.assistant.sql_validator import SQLGlotValidator


def test_validator_valid_query():
    validator = SQLGlotValidator()
    sql = "SELECT origin_zone, COUNT(*) AS trip_count FROM taxi_trips GROUP BY origin_zone ORDER BY trip_count DESC LIMIT 10"
    res = validator.validate_sql(sql)
    assert res.is_valid is True
    assert "trip_count" in res.extracted_aliases


def test_validator_multi_statement_injection():
    validator = SQLGlotValidator()
    sql = "SELECT * FROM taxi_trips; DROP TABLE taxi_trips;"
    res = validator.validate_sql(sql)
    assert res.is_valid is False
    assert "injection" in res.error_message.lower() or "single-select" in res.error_message.lower()


def test_validator_prohibited_join():
    validator = SQLGlotValidator()
    sql = "SELECT a.origin_zone, b.dest_zone FROM taxi_trips a JOIN external_weather b ON a.pickup_date = b.date"
    res = validator.validate_sql(sql)
    assert res.is_valid is False
    assert "prohibited" in res.error_message.lower() or "join" in res.error_message.lower() or "relation" in res.error_message.lower()


def test_validator_prohibited_cte():
    validator = SQLGlotValidator()
    sql = "WITH summary AS (SELECT origin_borough FROM taxi_trips) SELECT * FROM summary"
    res = validator.validate_sql(sql)
    assert res.is_valid is False
    assert "prohibited" in res.error_message.lower() or "with" in res.error_message.lower() or "relation" in res.error_message.lower()



def test_validator_unwhitelisted_function():
    validator = SQLGlotValidator()
    sql = "SELECT UNKNOWN_HACK_FUNC(base_fare) FROM taxi_trips"
    res = validator.validate_sql(sql)
    assert res.is_valid is False
    assert "unwhitelisted" in res.error_message.lower() or "unsafe" in res.error_message.lower()


def test_validator_invalid_column():
    validator = SQLGlotValidator()
    sql = "SELECT non_existent_column_abc FROM taxi_trips"
    res = validator.validate_sql(sql)
    assert res.is_valid is False
    assert "invalid" in res.error_message.lower() or "column" in res.error_message.lower()
