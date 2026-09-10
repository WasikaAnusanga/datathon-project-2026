"""
src/data/make_splits.py
-----------------------
Loads the enriched 12-month taxi dataset using fastparquet,
creates chronological splits (Train: M1-M8, Val: M9-M10, Test: M11-M12),
and computes historical lookup statistics strictly from the Training split.
"""

import os
import gc
import joblib
import numpy as np
import fastparquet

PROCESSED_PARQUET = "data/processed/Urban_Flow_Analytics_Taxi_Clean_Enriched_12Month.parquet"
SPLITS_DIR = "data/splits"
MODELS_DIR = "models"

def main():
    print("=" * 70)
    print("STEP 1: Generating Chronological Train / Val / Test Splits")
    print("=" * 70)
    
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    train_path = os.path.join(SPLITS_DIR, "train.parquet")
    val_path = os.path.join(SPLITS_DIR, "val.parquet")
    test_path = os.path.join(SPLITS_DIR, "test.parquet")
    stats_path = os.path.join(MODELS_DIR, "historical_od_stats.joblib")
    
    print(f"Reading enriched parquet metadata from {PROCESSED_PARQUET}...")
    pf = fastparquet.ParquetFile(PROCESSED_PARQUET)
    print(f"Total dataset rows: {pf.count():,}")
    
    # 1. Filter Train Split (Months 1 through 8: April 2025 - Nov 2025)
    print("\nExtracting Train split (Months 1–8)...")
    train_df = pf.to_pandas(filters=[('pickup_month', '<=', 8)])
    print(f"Train split shape: {train_df.shape} ({len(train_df):,} rows)")
    
    print(f"Saving Train split to {train_path}...")
    fastparquet.write(train_path, train_df, compression="SNAPPY")
    
    # 2. Compute Leakage-Safe Historical Lookup Statistics strictly on Train split
    print("\nComputing historical OD lookup statistics strictly on Train split (M1-M8)...")
    od_stats = train_df.groupby("od_pair").agg(
        od_median_fare=("base_fare", "median"),
        od_mean_fare=("base_fare", "mean"),
        od_median_distance=("distance_miles", "median"),
        od_median_duration=("trip_duration_minutes", "median"),
        od_trip_count=("base_fare", "count")
    ).reset_index()
    
    rare_mask = od_stats["od_trip_count"] < 5
    od_stats.loc[rare_mask, "od_median_fare"] = np.nan
    od_stats.loc[rare_mask, "od_median_distance"] = np.nan
    od_stats.loc[rare_mask, "od_median_duration"] = np.nan
    
    global_stats = {
        "global_median_fare": float(train_df["base_fare"].median()),
        "global_mean_fare": float(train_df["base_fare"].mean()),
        "global_median_distance": float(train_df["distance_miles"].median()),
        "global_median_duration": float(train_df["trip_duration_minutes"].median()),
        "od_stats_df": od_stats
    }
    
    print(f"Global Train Median Base Fare: ${global_stats['global_median_fare']:.2f}")
    print(f"Global Train Mean Base Fare:   ${global_stats['global_mean_fare']:.2f}")
    print(f"Unique OD routes in Train:    {len(od_stats):,}")
    
    joblib.dump(global_stats, stats_path)
    print(f"Saved historical lookup statistics to {stats_path}")
    
    del train_df
    gc.collect()
    
    # 3. Filter Validation Split (Months 9 & 10: Dec 2025 - Jan 2026)
    print("\nExtracting Validation split (Months 9–10)...")
    val_df = pf.to_pandas(filters=[('pickup_month', 'in', [9, 10])])
    print(f"Val split shape: {val_df.shape} ({len(val_df):,} rows)")
    print(f"Saving Val split to {val_path}...")
    fastparquet.write(val_path, val_df, compression="SNAPPY")
    del val_df
    gc.collect()
    
    # 4. Filter Test Split (Months 11 & 12: Feb 2026 - Mar 2026)
    print("\nExtracting Test split (Months 11–12)...")
    test_df = pf.to_pandas(filters=[('pickup_month', 'in', [11, 12])])
    print(f"Test split shape: {test_df.shape} ({len(test_df):,} rows)")
    print(f"Saving Test split to {test_path}...")
    fastparquet.write(test_path, test_df, compression="SNAPPY")
    del test_df
    gc.collect()
    
    print("\n" + "=" * 70)
    print("SUCCESS: Train, Val, Test splits and historical stats created successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
