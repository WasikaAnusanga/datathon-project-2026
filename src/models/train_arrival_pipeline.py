"""Train the leakage-safe on-time arrival estimator."""

from __future__ import annotations

import os
import sys

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.abspath("."))

from src.models.arrival_model import (  # noqa: E402
    ARRIVAL_FEATURES,
    ArrivalEstimator,
    build_historical_duration_stats,
    chronological_split,
    evaluate_arrival_predictions,
    evaluate_arrival_segments,
    LEAKAGE_COLUMNS,
    prepare_training_frame,
)


DATA_PATHS = [
    "data/processed/Urban_Flow_Analytics_Taxi_Trip_Time.parquet",
    "data/processed/Urban_Flow_Analytics_Taxi_Clean_Enriched_12Month.parquet",
    "data/processed/Urban_Flow_Analytics_Taxi_Clean_12Month.parquet",
    "data/processed/taxi_combined_12months.parquet",
]

REQUIRED_COLUMNS = [
    "pickup_timestamp",
    "dropoff_timestamp",
    "provider_code",
    "rider_count",
    "rate_class_id",
    "origin_loc_id",
    "dest_loc_id",
]

TRIP_TIME_MODEL_MAX_MINUTES = 180.0


def load_source(max_rows_per_split: int = 500_000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load chronological samples without requiring the full dataset in memory."""
    path = next((candidate for candidate in DATA_PATHS if os.path.exists(candidate)), None)
    if path is None:
        raise FileNotFoundError(
            "No cleaned Parquet source found. Run notebook 01 through the save-output step first."
        )

    available = pq.ParquetFile(path).schema.names
    columns = [column for column in REQUIRED_COLUMNS if column in available]
    columns += [
        column for column in [
            "is_cross_borough",
            "is_airport_trip",
            "origin_borough",
            "dest_borough",
            "origin_zone",
            "dest_zone",
        ]
        if column in available
    ]
    split_filters = [
        [("pickup_timestamp", "<", pd.Timestamp("2025-12-01"))],
        [
            ("pickup_timestamp", ">=", pd.Timestamp("2025-12-01")),
            ("pickup_timestamp", "<", pd.Timestamp("2026-02-01")),
        ],
        [("pickup_timestamp", ">=", pd.Timestamp("2026-02-01"))],
    ]
    samples = []
    for filters in split_filters:
        sample = pd.read_parquet(
            path,
            columns=columns,
            filters=filters,
            engine="pyarrow",
        )
        samples.append(sample.sort_values("pickup_timestamp").head(max_rows_per_split).copy())
    return tuple(samples)


def run_training(max_rows_per_split: int = 500_000) -> dict:
    train_raw, validation_raw, test_raw = load_source(max_rows_per_split)
    historical_stats = build_historical_duration_stats(
        train_raw,
        max_duration_minutes=TRIP_TIME_MODEL_MAX_MINUTES,
    )
    X_train, y_train = prepare_training_frame(
        train_raw,
        max_duration_minutes=TRIP_TIME_MODEL_MAX_MINUTES,
        historical_stats=historical_stats,
    )
    X_validation, y_validation = prepare_training_frame(
        validation_raw,
        max_duration_minutes=TRIP_TIME_MODEL_MAX_MINUTES,
        historical_stats=historical_stats,
    )
    X_test, y_test = prepare_training_frame(
        test_raw,
        max_duration_minutes=TRIP_TIME_MODEL_MAX_MINUTES,
        historical_stats=historical_stats,
    )

    print("Leakage audit:")
    print(f"Features used: {ARRIVAL_FEATURES}")
    assert not set(ARRIVAL_FEATURES) & LEAKAGE_COLUMNS
    print(f"Model duration policy: <= {TRIP_TIME_MODEL_MAX_MINUTES:.0f} minutes")
    print("dropoff_timestamp in X: False")
    print("post-pickup fields in X: False")
    print("Model: LightGBM gradient-boosted tree regression")
    print("Split: chronological train -> validation -> out-of-time test")

    baseline_prediction = y_train.median()
    baseline_metrics = evaluate_arrival_predictions(
        y_validation,
        [baseline_prediction] * len(y_validation),
    )

    od_baseline = y_validation.index.to_series().map(
        validation_raw.loc[y_validation.index, "origin_loc_id"].astype("int32") * 1000
        + validation_raw.loc[y_validation.index, "dest_loc_id"].astype("int32")
    ).map(historical_stats["od"]).fillna(historical_stats["global_median"])
    od_baseline_metrics = evaluate_arrival_predictions(y_validation, od_baseline)

    estimator = ArrivalEstimator.fit(
        X_train,
        y_train,
        X_validation,
        y_validation,
        historical_stats=historical_stats,
    )
    validation_metrics = evaluate_arrival_predictions(
        y_validation,
        estimator.predict_minutes(validation_raw.loc[X_validation.index]),
    )
    test_metrics = evaluate_arrival_predictions(
        y_test,
        estimator.predict_minutes(test_raw.loc[X_test.index]),
    )
    test_predictions = estimator.predict_minutes(test_raw.loc[X_test.index])
    segment_metrics = evaluate_arrival_segments(
        test_raw.loc[X_test.index],
        y_test,
        test_predictions,
    )

    estimator.save()
    print(f"Median baseline validation: {baseline_metrics}")
    print(f"Training-only OD median validation: {od_baseline_metrics}")
    print(f"LightGBM validation: {validation_metrics}")
    print(f"LightGBM out-of-time test: {test_metrics}")
    print("Out-of-time segment metrics:")
    print(segment_metrics.to_string(index=False))
    print("Example passenger ETA:")
    example = test_raw.loc[X_test.index].iloc[[0]].copy()
    example["predicted_duration_minutes"] = test_predictions[0]
    example["estimated_arrival_timestamp"] = estimator.predict_arrival(example).iloc[0]
    print(example[["pickup_timestamp", "predicted_duration_minutes", "estimated_arrival_timestamp"]].to_string(index=False))
    print("Saved model: models/arrival_time_estimator.pkl")
    return {
        "baseline_validation": baseline_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "segments": segment_metrics,
    }


if __name__ == "__main__":
    run_training()
