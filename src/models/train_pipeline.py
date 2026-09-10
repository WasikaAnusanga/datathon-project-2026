"""
src/models/train_pipeline.py
-----------------------------
Executes model training, baseline benchmarking, evaluation metric generation,
error analysis, and pipeline serialization for Upfront Base Fare Prediction.
Exports model artifact in .pkl format for competition submission.
"""

import os
import sys
import time
import pandas as pd
import fastparquet

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from src.features.fare_features import (
    load_historical_stats,
    get_feature_matrices
)
from src.models.fare_model import (
    GlobalMedianBaseline,
    ODMedianBaseline,
    train_ridge_baseline,
    train_lightgbm_model,
    train_xgboost_model,
    evaluate_predictions,
    UpfrontFarePipeline
)

def load_split_subset(parquet_path, sample_size=500000):
    """Fast memory-optimized loader loading only pre-trip features up to sample_size rows."""
    pf = fastparquet.ParquetFile(parquet_path)
    cols = ["provider_code", "rider_count", "distance_miles", "rate_class_id",
            "origin_loc_id", "dest_loc_id", "pickup_timestamp", "base_fare", "trip_duration_minutes",
            "is_cross_borough", "is_airport_trip", "pickup_hour", "pickup_dayofweek", "pickup_month",
            "is_weekend", "is_morning_rush", "is_evening_rush", "is_late_night",
            "sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek", "od_pair"]
    
    available_cols = [c for c in cols if c in pf.columns]
    
    df_list = []
    rows_acc = 0
    for chunk in pf.iter_row_groups(columns=available_cols):
        df_list.append(chunk)
        rows_acc += len(chunk)
        if sample_size and rows_acc >= sample_size:
            break
            
    df = pd.concat(df_list, ignore_index=True)
    if sample_size and len(df) > sample_size:
        df = df.iloc[:sample_size]
    return df

def run_experiment_pipeline(sample_size=500000):
    print("=" * 80)
    print(f"UPFRONT BASE FARE PREDICTION PIPELINE (Sample size: {sample_size:,} per split)")
    print("=" * 80)
    
    # 1. Load historical training lookup stats
    print("\n1. Loading historical training lookup stats...")
    stats_path = "models/historical_od_stats.joblib"
    historical_stats = load_historical_stats(stats_path)
    print(f"Loaded global median base fare: ${historical_stats['global_median_fare']:.2f}")
    
    # 2. Fast Ingestion of Train and Validation splits
    train_split_path = "data/splits/train.parquet"
    val_split_path = "data/splits/val.parquet"
    test_split_path = "data/splits/test.parquet"
    
    print("\n2. Ingesting Train and Validation split subsets...")
    t0 = time.time()
    train_raw = load_split_subset(train_split_path, sample_size)
    val_raw = load_split_subset(val_split_path, sample_size)
    print(f"Train subset shape: {train_raw.shape} (Loaded in {time.time()-t0:.1f}s)")
    print(f"Val subset shape:   {val_raw.shape}")
    
    # 3. Prepare Pre-Trip Features
    print("\n3. Enriching Pre-Trip features with leakage-safe lookup stats...")
    X_train, y_train = get_feature_matrices(train_raw, historical_stats)
    X_val, y_val = get_feature_matrices(val_raw, historical_stats)
    
    print(f"Pre-trip feature count: {X_train.shape[1]}")
    
    # 4. Model Training & Evaluation Benchmark
    metrics_list = []
    
    # --- Baseline 1: Global Median ---
    print("\n--- Baseline 1: Global Median ---")
    b1 = GlobalMedianBaseline().fit(X_train, y_train)
    p1_val = b1.predict(X_val)
    m1 = evaluate_predictions(y_val, p1_val, "Global Median Baseline", "Validation")
    metrics_list.append(m1)
    print(f"Val MAE: ${m1['MAE']:.4f} | RMSE: ${m1['RMSE']:.4f} | R2: {m1['R2']:.4f}")
    
    # --- Baseline 2: OD-Pair Historical Median ---
    print("\n--- Baseline 2: OD Median Baseline ---")
    b2 = ODMedianBaseline(historical_stats).fit(X_train, y_train)
    p2_val = b2.predict(X_val)
    m2 = evaluate_predictions(y_val, p2_val, "OD Median Baseline", "Validation")
    metrics_list.append(m2)
    print(f"Val MAE: ${m2['MAE']:.4f} | RMSE: ${m2['RMSE']:.4f} | R2: {m2['R2']:.4f}")
    
    # --- Baseline 3: Ridge Linear Regression ---
    print("\n--- Baseline 3: Ridge Regression ---")
    b3 = train_ridge_baseline(X_train, y_train)
    p3_val = b3.predict(X_val.fillna(0.0))
    m3 = evaluate_predictions(y_val, p3_val, "Ridge Regression", "Validation")
    metrics_list.append(m3)
    print(f"Val MAE: ${m3['MAE']:.4f} | RMSE: ${m3['RMSE']:.4f} | R2: {m3['R2']:.4f}")
    
    # --- Candidate 1: LightGBM Regressor ---
    print("\n--- Candidate 1: LightGBM Regressor ---")
    t0 = time.time()
    lgb_model = train_lightgbm_model(X_train, y_train, X_val, y_val)
    lgb_train_time = time.time() - t0
    p_lgb_val = lgb_model.predict(X_val)
    m_lgb = evaluate_predictions(y_val, p_lgb_val, "LightGBM Regressor", "Validation")
    metrics_list.append(m_lgb)
    print(f"Val MAE: ${m_lgb['MAE']:.4f} | RMSE: ${m_lgb['RMSE']:.4f} | R2: {m_lgb['R2']:.4f} (Trained in {lgb_train_time:.1f}s)")
    
    # --- Candidate 2: XGBoost Regressor ---
    print("\n--- Candidate 2: XGBoost Regressor ---")
    t0 = time.time()
    xgb_model = train_xgboost_model(X_train, y_train, X_val, y_val)
    xgb_train_time = time.time() - t0
    p_xgb_val = xgb_model.predict(X_val)
    m_xgb = evaluate_predictions(y_val, p_xgb_val, "XGBoost Regressor", "Validation")
    metrics_list.append(m_xgb)
    print(f"Val MAE: ${m_xgb['MAE']:.4f} | RMSE: ${m_xgb['RMSE']:.4f} | R2: {m_xgb['R2']:.4f} (Trained in {xgb_train_time:.1f}s)")
    
    # Compare models and select best
    metrics_df = pd.DataFrame(metrics_list)
    print("\n" + "=" * 80)
    print("MODEL VALIDATION COMPARISON TABLE")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    
    best_row = metrics_df.sort_values("MAE").iloc[0]
    print(f"\nBest Performing Model: {best_row['Model']} (Validation MAE: ${best_row['MAE']:.4f})")
    
    if best_row['Model'] == 'LightGBM Regressor':
        best_model = lgb_model
    elif best_row['Model'] == 'XGBoost Regressor':
        best_model = xgb_model
    else:
        best_model = b3
        
    # 5. Out-of-Time Test Set Evaluation
    print("\n5. Out-of-Time Test Set Evaluation (Months 11–12)...")
    test_raw = load_split_subset(test_split_path, sample_size)
    X_test, y_test = get_feature_matrices(test_raw, historical_stats)
    p_best_test = best_model.predict(X_test)
    m_test = evaluate_predictions(y_test, p_best_test, best_row['Model'], "Test (Out-of-Time)")
    
    print("\n" + "=" * 80)
    print("FINAL TEST SET BENCHMARK PERFORMANCE")
    print("=" * 80)
    print(f"Model    : {m_test['Model']}")
    print(f"Test MAE : ${m_test['MAE']:.4f}")
    print(f"Test RMSE: ${m_test['RMSE']:.4f}")
    print(f"Test R²  : {m_test['R2']:.4f}")
    print("=" * 80)
    
    # 6. Save Final Inference Pipeline in .pkl format for competition submission
    pipeline = UpfrontFarePipeline(best_model, historical_stats)
    pipeline.save("models/fare_prediction_pipeline.pkl")
    
    # 7. Verify single-row synthetic payload prediction
    print("\n7. Verifying single-row synthetic payload inference...")
    sample_request = {
        'provider_code': 1,
        'rider_count': 1,
        'distance_miles': 5.2,
        'rate_class_id': 1,
        'origin_loc_id': 138, # LaGuardia Airport area
        'dest_loc_id': 230,   # Times Square / Midtown
        'pickup_timestamp': '2026-04-15 17:30:00' # Evening rush
    }
    
    predicted_fare = pipeline.predict_single(sample_request)
    print(f"Sample Trip Request: Distance {sample_request['distance_miles']} miles, Zone {sample_request['origin_loc_id']} -> {sample_request['dest_loc_id']}")
    print(f"Predicted Upfront Base Fare: ${predicted_fare:.2f}")
    
    return metrics_df, pipeline

if __name__ == "__main__":
    run_experiment_pipeline(sample_size=500000)
