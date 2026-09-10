"""
src/features/fare_features.py
------------------------------
Defines strict pre-trip feature list and feature transformation functions.
Guarantees zero data leakage from post-trip attributes.
"""

import os
import joblib
import pandas as pd
import numpy as np

# Strict Pre-Trip Feature List (No post-trip fees, tips, tolls, duration or settlement methods)
PRE_TRIP_FEATURES = [
    "provider_code",
    "rider_count",
    "distance_miles",
    "rate_class_id",
    "origin_loc_id",
    "dest_loc_id",
    "is_cross_borough",
    "is_airport_trip",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "is_weekend",
    "is_morning_rush",
    "is_evening_rush",
    "is_late_night",
    "sin_hour",
    "cos_hour",
    "sin_dayofweek",
    "cos_dayofweek",
    "od_pair",
    "od_median_fare",
    "od_median_distance",
    "od_median_duration"
]

CATEGORICAL_FEATURES = [
    "provider_code",
    "rate_class_id",
    "origin_loc_id",
    "dest_loc_id"
]

EXCLUDED_POST_TRIP_LEAKAGE_COLS = [
    "dropoff_timestamp",
    "trip_duration_seconds",
    "trip_duration_minutes",
    "trip_duration_hours",
    "speed_mph",
    "driver_tip_payment",
    "toll_total",
    "surcharge_misc",
    "transit_tax",
    "service_improvement_fee",
    "charge_total",
    "zone_congestion_fee",
    "Airport_fee",
    "congestion_relief_fee",
    "fare_settlement_method",
    "calculated_charge_total",
    "charge_difference",
    "flag_charge_mismatch",
    "flag_negative_fare",
    "flag_zero_distance_nonzero_fare",
    "flag_zero_riders",
    "flag_dropoff_before_pickup",
    "flag_unrealistic_speed",
    "any_required_anomaly",
    "number_of_required_anomalies"
]

def load_historical_stats(stats_path="models/historical_od_stats.joblib"):
    """Loads training-set historical OD lookup statistics."""
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Historical stats file not found at {stats_path}. Run src/data/make_splits.py first.")
    return joblib.load(stats_path)

def prepare_pre_trip_dataframe(df, historical_stats):
    """
    Enriches a dataframe with pre-trip temporal, spatial, and leakage-safe
    historical route statistics derived strictly from the training set.
    """
    df = df.copy()
    
    # 1. Guarantee od_pair calculation if missing
    if "od_pair" not in df.columns:
        df["od_pair"] = df["origin_loc_id"].astype(np.int32) * 1000 + df["dest_loc_id"].astype(np.int32)
        
    # 2. Map historical OD lookup statistics (computed on Train set)
    od_stats_df = historical_stats["od_stats_df"]
    
    merged = pd.merge(
        df,
        od_stats_df[["od_pair", "od_median_fare", "od_median_distance", "od_median_duration"]],
        on="od_pair",
        how="left"
    )
    
    # Fill missing OD pairs with global training fallbacks
    global_fare = historical_stats["global_median_fare"]
    global_dist = historical_stats["global_median_distance"]
    global_dur = historical_stats["global_median_duration"]
    
    merged["od_median_fare"] = merged["od_median_fare"].fillna(global_fare).astype(np.float32)
    merged["od_median_distance"] = merged["od_median_distance"].fillna(merged["distance_miles"].fillna(global_dist)).astype(np.float32)
    merged["od_median_duration"] = merged["od_median_duration"].fillna(global_dur).astype(np.float32)
    
    for col in ["provider_code", "rate_class_id", "pickup_hour", "pickup_dayofweek", "pickup_month",
                "is_weekend", "is_morning_rush", "is_evening_rush", "is_late_night", "is_cross_borough", "is_airport_trip"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(np.int32)
            
    for col in ["rider_count", "distance_miles", "sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0).astype(np.float32)
            
    return merged

def get_feature_matrices(df, historical_stats, target_col="base_fare"):
    """
    Extracts pre-trip feature matrix X and target y.
    Returns (X, y).
    """
    enriched_df = prepare_pre_trip_dataframe(df, historical_stats)
    X = enriched_df[PRE_TRIP_FEATURES]
    
    if target_col in enriched_df.columns:
        y = enriched_df[target_col].astype(np.float32)
        return X, y
    return X, None
