"""
src/models/fare_model.py
------------------------
Model training, baseline evaluation, error analysis, and pipeline serialization for
Upfront Base Fare Prediction. Supports both .pkl and .joblib formats for competition submission.
"""

import os
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb

from src.features.fare_features import (
    PRE_TRIP_FEATURES,
    CATEGORICAL_FEATURES,
    prepare_pre_trip_dataframe
)

class GlobalMedianBaseline:
    """Baseline 1: Predicts global training median base fare."""
    def __init__(self):
        self.median_val = 0.0

    def fit(self, X, y):
        self.median_val = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.median_val, dtype=np.float32)

class ODMedianBaseline:
    """Baseline 2: Predicts historical median base fare for each OD route."""
    def __init__(self, global_stats):
        self.global_stats = global_stats

    def fit(self, X, y):
        return self

    def predict(self, X):
        if "od_median_fare" in X.columns:
            return X["od_median_fare"].fillna(self.global_stats["global_median_fare"]).to_numpy(dtype=np.float32)
        return np.full(len(X), self.global_stats["global_median_fare"], dtype=np.float32)

def evaluate_predictions(y_true, y_pred, model_name="Model", dataset_name="Val"):
    """Calculates MAE, RMSE, R2 metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        "Model": model_name,
        "Dataset": dataset_name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }
    return metrics

def train_ridge_baseline(X_train, y_train):
    """Trains a Ridge Linear Regression baseline."""
    print("Training Ridge Regression baseline...")
    X_train_clean = X_train.fillna(0.0)
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train_clean, y_train)
    return model

def train_lightgbm_model(X_train, y_train, X_val=None, y_val=None):
    """Trains a LightGBM Regressor optimized for tabular fare prediction."""
    print("Training LightGBM Regressor...")
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    
    if X_val is not None and y_val is not None:
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
            categorical_feature=[c for c in CATEGORICAL_FEATURES if c in X_train.columns]
        )
    else:
        model.fit(
            X_train, y_train,
            categorical_feature=[c for c in CATEGORICAL_FEATURES if c in X_train.columns]
        )
        
    return model

def train_xgboost_model(X_train, y_train, X_val=None, y_val=None):
    """Trains an XGBoost Regressor."""
    print("Training XGBoost Regressor...")
    
    params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'n_estimators': 800,
        'learning_rate': 0.05,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    
    model = xgb.XGBRegressor(**params)
    
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train)
        
    return model

class UpfrontFarePipeline:
    """
    Unified end-to-end inference pipeline for Upfront Base Fare Amount Prediction.
    Bundles pre-trip feature transformations, historical lookup statistics, and the trained model.
    Supports serialization to both .pkl and .joblib formats.
    """
    def __init__(self, model, historical_stats):
        self.model = model
        self.historical_stats = historical_stats
        self.feature_names = PRE_TRIP_FEATURES

    def predict_single(self, trip_request: dict) -> float:
        """
        Predicts base_fare for a single pre-trip request dictionary.
        """
        df = pd.DataFrame([trip_request])
        
        if "pickup_timestamp" in df.columns:
            df["pickup_timestamp"] = pd.to_datetime(df["pickup_timestamp"])
            df["pickup_hour"] = df["pickup_timestamp"].dt.hour.astype(np.int32)
            df["pickup_dayofweek"] = df["pickup_timestamp"].dt.dayofweek.astype(np.int32)
            df["pickup_month"] = df["pickup_timestamp"].dt.month.astype(np.int32)
            df["is_weekend"] = (df["pickup_dayofweek"] >= 5).astype(np.int32)
            df["is_morning_rush"] = ((df["is_weekend"] == 0) & (df["pickup_hour"].between(7, 10))).astype(np.int32)
            df["is_evening_rush"] = ((df["is_weekend"] == 0) & (df["pickup_hour"].between(16, 20))).astype(np.int32)
            df["is_late_night"] = ((df["pickup_hour"] >= 23) | (df["pickup_hour"] <= 5)).astype(np.int32)
            
            df["sin_hour"] = np.sin(2 * np.pi * df["pickup_hour"] / 24.0).astype(np.float32)
            df["cos_hour"] = np.cos(2 * np.pi * df["pickup_hour"] / 24.0).astype(np.float32)
            df["sin_dayofweek"] = np.sin(2 * np.pi * df["pickup_dayofweek"] / 7.0).astype(np.float32)
            df["cos_dayofweek"] = np.cos(2 * np.pi * df["pickup_dayofweek"] / 7.0).astype(np.float32)
            
        if "is_cross_borough" not in df.columns:
            df["is_cross_borough"] = np.int32(0)
        if "is_airport_trip" not in df.columns:
            df["is_airport_trip"] = np.int32(1 if trip_request.get("rate_class_id") in [2, 3] else 0)
            
        enriched = prepare_pre_trip_dataframe(df, self.historical_stats)
        X = enriched[PRE_TRIP_FEATURES]
        
        pred = self.model.predict(X)[0]
        return max(2.50, float(pred))

    def save(self, filepath="models/fare_prediction_pipeline.pkl"):
        """Saves the pipeline to disk in .pkl or .joblib format."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if filepath.endswith(".pkl"):
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
        else:
            joblib.dump(self, filepath)
        print(f"UpfrontFarePipeline saved to {filepath}")

    @classmethod
    def load(cls, filepath="models/fare_prediction_pipeline.pkl"):
        """Loads the pipeline from disk from .pkl or .joblib format."""
        if not os.path.exists(filepath):
            alt_path = filepath.replace(".pkl", ".joblib") if filepath.endswith(".pkl") else filepath.replace(".joblib", ".pkl")
            if os.path.exists(alt_path):
                filepath = alt_path
            else:
                raise FileNotFoundError(f"Pipeline file not found: {filepath}")
                
        if filepath.endswith(".pkl"):
            with open(filepath, "rb") as f:
                return pickle.load(f)
        return joblib.load(filepath)
