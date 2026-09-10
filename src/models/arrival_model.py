"""Leakage-safe on-time arrival estimation for UrbanFlow Analytics."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ARRIVAL_FEATURES = [
    "provider_code",
    "rider_count",
    "rate_class_id",
    "origin_loc_id",
    "dest_loc_id",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "is_weekend",
    "is_morning_rush",
    "is_evening_rush",
    "is_late_night",
    "hour_sin",
    "hour_cos",
    "sin_dayofweek",
    "cos_dayofweek",
    "is_cross_borough",
    "is_airport_trip",
    "origin_borough_code",
    "dest_borough_code",
    "origin_service_zone_code",
    "dest_service_zone_code",
    "od_pair_code",
    "historical_od_median",
    "historical_od_hour_median",
    "historical_od_weekday_median",
]

POST_PICKUP_COLUMNS = {
    "distance_miles",
    "dropoff_timestamp",
    "trip_duration_seconds",
    "trip_duration_minutes",
    "trip_duration_hours",
    "speed_mph",
    "driver_tip_payment",
    "toll_total",
    "charge_total",
    "base_fare",
    "fare_settlement_method",
}

DESTINATION_KNOWN_BEFORE_DEPARTURE = True
LEAKAGE_COLUMNS = POST_PICKUP_COLUMNS | {"pickup_timestamp"}


def build_arrival_target(df: pd.DataFrame) -> pd.Series:
    """Build duration in minutes from timestamps; timestamps never enter X."""
    pickup = pd.to_datetime(df["pickup_timestamp"], errors="coerce")
    dropoff = pd.to_datetime(df["dropoff_timestamp"], errors="coerce")
    duration = (dropoff - pickup).dt.total_seconds().div(60)
    return duration.astype("float32")


def build_historical_duration_stats(
    df: pd.DataFrame,
    max_duration_minutes: float | None = None,
) -> dict:
    """Build duration lookups from one partition, normally the training data only."""
    target = build_arrival_target(df)
    valid = target.notna() & target.gt(0)
    if max_duration_minutes is not None:
        valid &= target.le(max_duration_minutes)

    work = pd.DataFrame({
        "duration": target.loc[valid],
        "origin_loc_id": df.loc[valid, "origin_loc_id"].astype("int32"),
        "dest_loc_id": df.loc[valid, "dest_loc_id"].astype("int32"),
        "pickup_hour": pd.to_datetime(
            df.loc[valid, "pickup_timestamp"], errors="coerce"
        ).dt.hour.astype("int8"),
        "pickup_dayofweek": pd.to_datetime(
            df.loc[valid, "pickup_timestamp"], errors="coerce"
        ).dt.dayofweek.astype("int8"),
    }).dropna()
    work["od_pair_code"] = (
        work["origin_loc_id"] * 1000 + work["dest_loc_id"]
    )
    return {
        "global_median": float(work["duration"].median()),
        "od": work.groupby("od_pair_code")["duration"].median(),
        "od_hour": work.groupby(["od_pair_code", "pickup_hour"])["duration"].median(),
        "od_weekday": work.groupby(
            ["od_pair_code", "pickup_dayofweek"]
        )["duration"].median(),
    }


def build_pre_trip_features(
    df: pd.DataFrame,
    historical_stats: dict | None = None,
) -> pd.DataFrame:
    """Create only features available at pickup time or earlier."""
    features = pd.DataFrame(index=df.index)
    numeric_defaults = {
        "provider_code": 0,
        "rider_count": np.nan,
        "rate_class_id": 99,
        "origin_loc_id": -1,
        "dest_loc_id": -1,
    }
    for column, default in numeric_defaults.items():
        features[column] = df[column] if column in df else default

    pickup = pd.to_datetime(df["pickup_timestamp"], errors="coerce")
    features["pickup_hour"] = pickup.dt.hour.astype("float32")
    features["pickup_dayofweek"] = pickup.dt.dayofweek.astype("float32")
    features["pickup_month"] = pickup.dt.month.astype("float32")
    features["is_weekend"] = (features["pickup_dayofweek"] >= 5).astype("int8")
    features["is_morning_rush"] = (
        features["pickup_hour"].between(7, 10)
    ).astype("int8")
    features["is_evening_rush"] = (
        features["pickup_hour"].between(16, 19)
    ).astype("int8")
    features["is_late_night"] = (
        (features["pickup_hour"] >= 23) | (features["pickup_hour"] <= 5)
    ).astype("int8")

    features["hour_sin"] = np.sin(2 * np.pi * features["pickup_hour"] / 24).astype("float32")
    features["hour_cos"] = np.cos(2 * np.pi * features["pickup_hour"] / 24).astype("float32")
    features["sin_dayofweek"] = np.sin(
        2 * np.pi * features["pickup_dayofweek"] / 7
    ).astype("float32")
    features["cos_dayofweek"] = np.cos(
        2 * np.pi * features["pickup_dayofweek"] / 7
    ).astype("float32")

    if "is_cross_borough" in df:
        features["is_cross_borough"] = df["is_cross_borough"].fillna(0).astype("int8")
    elif "origin_borough" in df and "dest_borough" in df:
        features["is_cross_borough"] = (
            df["origin_borough"] != df["dest_borough"]
        ).astype("int8")
    else:
        features["is_cross_borough"] = 0

    if "is_airport_trip" in df:
        features["is_airport_trip"] = df["is_airport_trip"].fillna(0).astype("int8")
    else:
        airport_rate = features["rate_class_id"].isin([2, 3])
        airport_text = pd.Series(False, index=df.index)
        for column in ("origin_zone", "dest_zone"):
            if column in df:
                airport_text |= df[column].astype("string").str.contains(
                    "airport", case=False, na=False
                )
        features["is_airport_trip"] = (airport_rate | airport_text).astype("int8")

    category_codes = {
        "origin_borough": ("origin_borough_code", [
            "Unknown", "Bronx", "Brooklyn", "EWR", "Manhattan", "Queens",
            "Staten Island",
        ]),
        "dest_borough": ("dest_borough_code", [
            "Unknown", "Bronx", "Brooklyn", "EWR", "Manhattan", "Queens",
            "Staten Island",
        ]),
        "origin_service_zone": ("origin_service_zone_code", [
            "Unknown", "Airports", "Boro Zone", "EWR", "Yellow Zone",
        ]),
        "dest_service_zone": ("dest_service_zone_code", [
            "Unknown", "Airports", "Boro Zone", "EWR", "Yellow Zone",
        ]),
    }
    for source_column, (output_column, categories) in category_codes.items():
        if source_column in df:
            values = df[source_column].astype("string").fillna("Unknown")
            features[output_column] = pd.Categorical(
                values,
                categories=categories,
            ).codes.astype("int8")
        else:
            features[output_column] = 0

    # Numeric encoding preserves the ordered OD relationship without using any
    # post-pickup measurement or unstable per-split string factorization.
    features["od_pair_code"] = (
        features["origin_loc_id"].fillna(-1).astype("int32") * 1000
        + features["dest_loc_id"].fillna(-1).astype("int32")
    )

    global_median = (
        historical_stats["global_median"]
        if historical_stats is not None
        else np.nan
    )
    features["historical_od_median"] = (
        features["od_pair_code"].map(historical_stats["od"])
        if historical_stats is not None
        else global_median
    )
    hour_key = pd.MultiIndex.from_arrays(
        [features["od_pair_code"], features["pickup_hour"]],
    )
    weekday_key = pd.MultiIndex.from_arrays(
        [features["od_pair_code"], features["pickup_dayofweek"]],
    )
    features["historical_od_hour_median"] = (
        pd.Series(historical_stats["od_hour"].reindex(hour_key).to_numpy(), index=df.index)
        if historical_stats is not None
        else global_median
    )
    features["historical_od_weekday_median"] = (
        pd.Series(
            historical_stats["od_weekday"].reindex(weekday_key).to_numpy(),
            index=df.index,
        )
        if historical_stats is not None
        else global_median
    )
    if historical_stats is not None:
        for column in [
            "historical_od_median",
            "historical_od_hour_median",
            "historical_od_weekday_median",
        ]:
            features[column] = features[column].fillna(global_median).astype("float32")

    return features[ARRIVAL_FEATURES]


def prepare_training_frame(
    df: pd.DataFrame,
    max_duration_minutes: float | None = None,
    historical_stats: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return valid target and pre-trip features with aligned rows.

    A duration cap is optional and task-specific. Extreme records remain in the
    audit dataset and should be investigated before enabling this filter.
    """
    target = build_arrival_target(df)
    valid = target.notna() & target.gt(0) & df["pickup_timestamp"].notna()
    if max_duration_minutes is not None:
        valid &= target.le(max_duration_minutes)
    X = build_pre_trip_features(df.loc[valid], historical_stats)
    y = target.loc[valid]
    return X, y


def chronological_split(
    df: pd.DataFrame,
    train_end: str = "2025-12-01",
    validation_end: str = "2026-02-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by pickup time; no future rows enter an earlier split."""
    pickup = pd.to_datetime(df["pickup_timestamp"], errors="coerce")
    train = df.loc[pickup < pd.Timestamp(train_end)]
    validation = df.loc[
        (pickup >= pd.Timestamp(train_end)) & (pickup < pd.Timestamp(validation_end))
    ]
    test = df.loc[pickup >= pd.Timestamp(validation_end)]
    return train, validation, test


def evaluate_arrival_predictions(y_true: Iterable[float], y_pred: Iterable[float]) -> dict:
    """Return minute-based regression metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.maximum(np.asarray(y_pred), 0)
    absolute_error = np.abs(y_true - y_pred)
    return {
        "MAE_minutes": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_minutes": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "within_2_minutes_pct": float((absolute_error <= 2).mean() * 100),
        "within_5_minutes_pct": float((absolute_error <= 5).mean() * 100),
        "within_10_minutes_pct": float((absolute_error <= 10).mean() * 100),
    }


def evaluate_arrival_segments(
    raw: pd.DataFrame,
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> pd.DataFrame:
    """Evaluate useful traffic and trip segments on an already filtered split."""
    y_true = pd.Series(np.asarray(y_true), index=raw.index)
    y_pred = pd.Series(np.maximum(np.asarray(y_pred), 0), index=raw.index)
    pickup = pd.to_datetime(raw["pickup_timestamp"], errors="coerce")
    hour = pickup.dt.hour
    weekday = pickup.dt.dayofweek < 5
    rush = hour.between(7, 10) | hour.between(16, 19)
    actual_duration = y_true
    segments = {
        "overall": pd.Series(True, index=raw.index),
        "rush_hour": rush,
        "non_rush_hour": ~rush,
        "weekday": weekday,
        "weekend": ~weekday,
        "short_trip_0_15m": actual_duration <= 15,
        "medium_trip_15_45m": actual_duration.gt(15) & actual_duration.le(45),
        "long_trip_over_45m": actual_duration > 45,
    }
    for column, label in [("origin_borough", "origin"), ("dest_borough", "destination")]:
        if column in raw:
            for value in raw[column].dropna().astype(str).unique():
                segments[f"{label}_{value}"] = raw[column].astype("string").eq(value)

    rows = []
    for name, mask in segments.items():
        if int(mask.sum()) == 0:
            continue
        metrics = evaluate_arrival_predictions(y_true[mask], y_pred[mask])
        rows.append({"segment": name, "rows": int(mask.sum()), **metrics})
    return pd.DataFrame(rows)


@dataclass
class ArrivalEstimator:
    """Median baseline plus LightGBM estimator and arrival-time inference."""

    model: lgb.LGBMRegressor
    duration_median: float
    historical_stats: dict | None = None
    feature_names: tuple[str, ...] = tuple(ARRIVAL_FEATURES)

    @classmethod
    def fit(
        cls,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame | None = None,
        y_validation: pd.Series | None = None,
        historical_stats: dict | None = None,
    ) -> "ArrivalEstimator":
        model = lgb.LGBMRegressor(
            objective="regression_l1",
            metric="mae",
            n_estimators=1200,
            learning_rate=0.04,
            num_leaves=63,
            max_depth=10,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        fit_kwargs = {}
        if X_validation is not None and y_validation is not None:
            fit_kwargs["eval_set"] = [(X_validation, y_validation)]
            fit_kwargs["callbacks"] = [lgb.early_stopping(75, verbose=False)]
        model.fit(X_train, y_train, **fit_kwargs)
        return cls(
            model=model,
            duration_median=float(y_train.median()),
            historical_stats=historical_stats,
        )

    def predict_minutes(self, requests: pd.DataFrame) -> np.ndarray:
        return np.maximum(
            self.model.predict(build_pre_trip_features(requests, self.historical_stats)),
            0,
        )

    def predict_arrival(self, requests: pd.DataFrame) -> pd.Series:
        pickup = pd.to_datetime(requests["pickup_timestamp"], errors="raise")
        minutes = self.predict_minutes(requests)
        return pd.Series(
            pickup + pd.to_timedelta(minutes, unit="m"),
            index=requests.index,
            name="estimated_arrival_timestamp",
        )

    def save(self, filepath: str = "models/arrival_time_estimator.pkl") -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, filepath: str = "models/arrival_time_estimator.pkl") -> "ArrivalEstimator":
        with open(filepath, "rb") as handle:
            return pickle.load(handle)
