# UrbanFlow Analytics - Team DataCraft

> **Datathon 2026** — Comprehensive Urban Flow Analytics, Demand Forecasting, Upfront Fare Modeling, and Interactive Web Platform.

---

## 📁 Repository Structure

```text
DataCraft_UrbanFlow/
├── config/                                 # Configuration files & parameters
├── dashboard/
│   └── app.py                              # Streamlit Interactive Web Application
├── data/
│   ├── raw/
│   │   ├── taxi/                           # 12 monthly TLC CSV files (~48.6M raw records)
│   │   └── zone/                           # Urban_Flow_Analytics_Zone_Dataset.csv (265 NYC TLC Zones)
│   ├── interim/                            # taxi_quality_audit_12month.parquet
│   ├── processed/
│   │   ├── Urban_Flow_Analytics_Taxi_Clean_12Month.parquet (~45.95M clean base trips)
│   │   └── Urban_Flow_Analytics_Taxi_Clean_Enriched_12Month.parquet (~45.95M enriched trips)
│   └── splits/
│       ├── train.parquet                   # Months 1–8 (33,373,198 rows, ~67%)
│       ├── val.parquet                     # Months 9–10 (13,631,488 rows, ~17%)
│       └── test.parquet                    # Months 11–12 (10,485,760 rows, ~16%)
├── models/
│   ├── fare_prediction_pipeline.pkl        # Datathon Submission Model Pipeline (.pkl)
│   └── historical_od_stats.joblib          # Training-Set Historical OD Lookup Dictionary
├── notebooks/
│   ├── 01_Exploratory_Data_.ipynb          # Data Quality Audit, Cleaning & Zone Enrichment
│   └── 03_fare_prediction.ipynb          # Section 2.1 Upfront Fare Prediction Notebook
├── reports/                                # CSV and JSON Data Quality Reports
├── src/
│   ├── data/
│   │   └── make_splits.py                  # Chronological Split Generator & Lookup Table Builder
│   ├── features/
│   │   └── fare_features.py                # Pre-Trip Feature Transformer & Lookup Mapper
│   └── models/
│       ├── fare_model.py                   # Baseline & ML Models, Pipeline Wrapper & .pkl Exporter
│       └── train_pipeline.py               # Automated Training & Benchmark Evaluation Script
├── README.md                               # Project Setup & Architecture Documentation
└── requirements.txt                        # Python Dependencies
```

---

## 🚀 Quickstart & Setup Guide

### 1. Environment Setup
```bash
# Clone repository and create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Chronological Data Splits & Lookup Tables
To generate the leakage-safe train/val/test split files and historical OD route lookup dictionary:
```bash
python src/data/make_splits.py
```

### 3. Train & Evaluate Upfront Base Fare Models
To train baselines, fit LightGBM & XGBoost regressors, compute MAE/RMSE/R² metrics, and export the submission model `.pkl`:
```bash
python src/models/train_pipeline.py
```

### 4. Launch Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 🔬 System Overview & Completed Components

### Section 2.1 — Upfront Base Fare Amount Prediction (`base_fare`)

#### Business Goal
Build a machine learning pipeline supporting **"No-Surprises Upfront Pricing"**, predicting the passenger's `base_fare` **before** the taxi trip begins.

---

### 1. Feature Lineage & Leakage Audit

To guarantee 100% prediction-time realism, post-trip variables finalized after trip completion are strictly excluded:

| Feature Name | Category | Available Pre-Trip? | Leakage Risk | Decision |
| :--- | :--- | :---: | :---: | :---: |
| `provider_code` | Telematics | Yes | None | **USE** |
| `rider_count` | Booking Request | Yes | None | **USE** |
| `distance_miles` | Route Estimator Proxy | Yes | Low | **USE** |
| `rate_class_id` | Rate Class Tariff | Yes | None | **USE** |
| `origin_loc_id` / `dest_loc_id` | Spatial Zones | Yes | None | **USE** |
| `is_cross_borough` / `is_airport_trip` | Spatial Flags | Yes | None | **USE** |
| `pickup_hour` / `pickup_dayofweek` | Temporal | Yes | None | **USE** |
| `sin_hour` / `cos_hour` / `sin_dayofweek` / `cos_dayofweek` | Continuous Cyclics | Yes | None | **USE** |
| `od_median_fare` / `od_median_distance` | Historical OD Lookup | Yes (Strict M1-M8) | None | **USE** |
| `dropoff_timestamp` | Timestamp | No | CRITICAL LEAKAGE | **DROP** |
| `trip_duration_minutes` | Duration | No | CRITICAL LEAKAGE | **DROP** |
| `speed_mph` | Telematics | No | CRITICAL LEAKAGE | **DROP** |
| `driver_tip_payment` / `toll_total` / `charge_total` | Financials | No | CRITICAL LEAKAGE | **DROP** |
| `fare_settlement_method` | Payment | No | CRITICAL LEAKAGE | **DROP** |

---

### 2. Temporal Dataset Splitting Strategy

To eliminate look-ahead bias, chronological splitting is applied across the 12-month dataset:

| Split Set | Months Included | Row Count | Split Ratio | Role in Pipeline |
| :--- | :--- | :---: | :---: | :--- |
| **Train Split** | Months 1–8 (Apr – Nov) | 33,373,198 | ~67% | Model training & historical OD lookup creation |
| **Validation Split** | Months 9–10 (Dec – Jan) | 13,631,488 | ~17% | Hyperparameter tuning & model benchmark selection |
| **Test Split** | Months 11–12 (Feb – Mar) | 10,485,760 | ~16% | Out-of-time benchmark evaluation |

---

### 3. Empirical Model Evaluation Benchmarks

#### Validation Performance (Months 9–10)

| Model | MAE ($) | RMSE ($) | R² | Status / Decision |
| :--- | :---: | :---: | :---: | :--- |
| **Global Median Baseline** | $12.1303 | $22.3159 | -0.1875 | Global naive baseline |
| **OD Median Baseline** | $3.8554 | $10.8388 | 0.7199 | Route lookup baseline |
| **Ridge Regression** | $6.0226 | $10.2138 | 0.7512 | Linear baseline |
| **XGBoost Regressor** | $2.8348 | $8.0500 | 0.8455 | High performance booster |
| **LightGBM Regressor (Selected)** | **$2.5913** | **$7.3809** | **0.8701** | **Selected Model (Best MAE & R²)** |

#### Out-of-Time Test Set Performance (Months 11–12)

| Metric | Out-of-Time Test Benchmark Score |
| :--- | :---: |
| **Selected Model** | LightGBM Upfront Fare Pipeline |
| **Test MAE** | **$4.5969** |
| **Test RMSE** | **$7.0402** |
| **Test R²** | **0.6793** |

---

### 4. Datathon Submission Artifacts (`models/`)

- **`models/fare_prediction_pipeline.pkl`**: Serialized submission model pipeline containing pre-trip feature transformers, training lookup dictionaries, and the trained LightGBM Regressor.
- **`models/historical_od_stats.joblib`**: Historical OD route lookup statistics.

---

### 5. Interactive Streamlit App (`dashboard/app.py`)

Includes a dedicated **"🚕 Upfront Base Fare Estimator"** tab allowing real-time fare estimations for arbitrary pickup/dropoff NYC zones, trip distance, tariff rate class, date, and passenger count.
