# UrbanFlow Analytics - Team DataCraft

> **Datathon 2026** - Comprehensive Urban Flow Analytics, Demand Forecasting, Fare & ETA Modeling, and Interactive Dashboard.

---

## 📁 Repository Structure

```text
DataCraft_UrbanFlow/
|-- data/
|   |-- raw/
|   |-- interim/
|   |-- processed/
|   `-- splits/
|-- notebooks/
|   |-- 01_data_understanding.ipynb
|   |-- 02_data_quality.ipynb
|   |-- 03_fare_prediction.ipynb
|   |-- 04_eta_prediction.ipynb
|   |-- 05_demand_forecasting.ipynb
|   |-- 06_hotspot_od_analysis.ipynb
|   `-- DataCraft_FinalNotebook.ipynb
|-- src/
|   |-- data/
|   |-- features/
|   |-- models/
|   |-- forecasting/
|   |-- analytics/
|   `-- assistant/
|-- models/
|-- dashboard/
|   `-- app.py
|-- reports/figures/
|-- config/
|-- requirements.txt
|-- README.md
`-- .gitignore
```

---

## 🚀 Getting Started

### 1. Setup Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 🔬 Data Pipeline & Quality Engineering (`notebooks/01_Exploratory_Data_Quality_Cleaning.ipynb`)

This section documents the foundational data cleaning, anomaly detection, enrichment, feature engineering, and dataset splitting pipelines executed in **Notebook 01**. Team members building downstream models (Fare Regression, ETA Prediction, Demand Forecasting) should read this carefully before training models.

---

### 1. Dataset Dimensions & Ingestion Summary
* **Source Files**: 12 monthly NYC Taxi & Limousine Commission (TLC) Parquet files + Taxi Zone Lookup table (`taxi_zone_lookup.csv`).
* **Raw Record Count**: **~48,599,839 rows** (~48.6 Million records across 12 months).
* **Final Clean/Enriched Count**: **~45,956,110 rows** (retaining **94.56%** of high-integrity real-world trips).
* **Removed Anomaly Rows**: **~2,643,729 rows** (**5.44%** discarded via strict operational audit criteria).

---

### 2. Data Quality Audit & Cleaning Decision Matrix

Each raw record underwent multi-point validation flags:

| Flag Name | Condition Detected | Action | Engineering Justification for Teammates |
| :--- | :--- | :--- | :--- |
| `flag_dropoff_before_pickup` | `dropoff_timestamp < pickup_timestamp` | **DROPPED** | Physically impossible negative durations caused by system clock resets or corrupt GPS transmissions. |
| `flag_negative_fare` | `base_fare < 0` or `charge_total < 0` | **DROPPED** | Chargebacks, voids, or driver correction penalties; not genuine passenger trips. |
| `flag_zero_riders` | `rider_count == 0` | **DROPPED** | Ghost trips or freight/package runs without passenger metadata. |
| `flag_unrealistic_speed` | Trip speed `> 80.0 mph` (calculated over distance & duration) | **DROPPED** | Unrealistic in NYC traffic; signals faulty GPS distance or inaccurate meter logging. |
| `flag_zero_distance_nonzero_fare` | `distance_miles == 0` & `base_fare > 0` | **RETAINED** | Valid operational trips (e.g. flat-rate waiting time, immediate cancellation with cancellation fee). Kept for accounting integrity. |
| *Exact Duplicate Rows* | All feature values identical across raw columns | **DROPPED** | Redundant message queue transmissions and re-posted telematics batches. |
| *Key Signature Duplicates* | Same pickup, dropoff, locations, distance & fare | **RETAINED** | Paired billing reconciliations/disputes; preserved for operational authenticity. |

---

### 3. Feature Engineering Specifications

All features were engineered with **zero temporal data leakage** (only using information available at or before dispatch/pickup time).

#### A. Pre-Trip Temporal Features
* `pickup_hour`: Hour of pickup `[0 - 23]` (`int8`).
* `pickup_dayofweek`: Day of week `[0 = Mon, 6 = Sun]` (`int8`).
* `pickup_month`: Calendar month `[1 - 12]` (`int8`).
* `is_weekend`: Binary flag `[0, 1]` indicating Saturday or Sunday (`int8`).
* `is_morning_rush`: Weekdays between 07:00 and 10:00 (`int8`).
* `is_evening_rush`: Weekdays between 16:00 and 20:00 (`int8`).
* `is_late_night`: High-risk/late-night hours between 23:00 and 05:00 (`int8`).
* `sin_hour` / `cos_hour`: Cyclic trigonometric transform ensuring smooth cyclical continuity between 23:00 and 00:00 (`float32`).
* `sin_dayofweek` / `cos_dayofweek`: Cyclic trigonometric transform of the weekly cycle (`float32`).

#### B. Spatial & Geographic Features
* `origin_loc_id` & `dest_loc_id`: NYC TLC Zone identifiers `[1 - 265]` (`int16`).
* `origin_borough` & `dest_borough`: Enriched via taxi zone metadata.
* `is_cross_borough`: Binary flag `[0, 1]` indicating whether pickup and dropoff happen in different boroughs (`int8`).
* `is_airport_trip`: Binary flag `[0, 1]` indicating JFK, LaGuardia, or Newark airport runs (`int8`).
* `od_pair`: Compact numerical encoding computed as `origin_loc_id * 1000 + dest_loc_id` (`int32`), avoiding multi-gigabyte string overhead.
* `od_avg_duration`: Historical baseline expected duration (in minutes) for that specific OD route. Uses distance heuristic fallback (`distance_miles * 3.5`) for rare routes (`float32`).

#### C. Target Variables for Modeling
* **Fare Model Target**: `base_fare` (and `charge_total` for total cost baselines).
* **ETA Model Target**: `trip_duration_minutes` (`float32`), calculated directly as:
  $$\text{trip\_duration\_minutes} = \frac{\text{dropoff\_timestamp} - \text{pickup\_timestamp}}{60 \text{ seconds}}$$
  *(Clipped at a minimum of 0.1 minutes to prevent divide-by-zero errors).*

---

### 4. Critical: Prevention of Target / Data Leakage

> [!WARNING]
> **DO NOT USE POST-TRIP COLUMNS AS MODEL FEATURES!**
> The following columns reflect charges, tolls, and events finalized **after** the trip concludes. If used for predicting upfront fares or ETAs, your models will artificially overfit and fail in production.

**Strictly Excluded Post-Trip Leakage Columns**:
* `fare_settlement_method` (payment type)
* `driver_tip_payment` (tips entered after trip completion)
* `charge_total` (includes post-trip tolls, congestion fees, and tips)
* `toll_total` (incurred while on the road)
* `surcharge_misc`
* `transit_tax`
* `service_improvement_fee`
* `zone_congestion_fee` / `congestion_relief_fee`

---

### 5. Recommended Pre-Trip Training Feature Set

When training your models, use this standardized feature array:

```python
PRE_TRIP_FEATURES = [
    # Trip Setup
    "provider_code",
    "rider_count",
    "distance_miles",
    "rate_class_id",
    # Spatial
    "origin_loc_id",
    "dest_loc_id",
    "is_cross_borough",
    "is_airport_trip",
    # Route Baseline
    "od_avg_duration",
    # Temporal
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "is_weekend",
    "is_morning_rush",
    "is_evening_rush",
    "is_late_night",
    # Continuous Cyclics
    "sin_hour",
    "cos_hour",
    "sin_dayofweek",
    "cos_dayofweek",
]
```

---

### 6. Processed Artifacts & Time-Based Train / Val / Test Splits

All outputs are saved as memory-efficient, Snappy-compressed Parquet files under `data/`:

#### A. Full Datasets
* `data/processed/Urban_Flow_Analytics_Taxi_Clean_12Month.parquet`: Full cleaned dataset (base fields).
* `data/processed/Urban_Flow_Analytics_Taxi_Clean_Enriched_12Month.parquet`: Full enriched dataset containing all pre-trip and cyclic features.

#### B. Time-Based Splits (`data/splits/`)
To strictly replicate production forecasting and prevent look-ahead bias, **time-based splitting** was applied instead of random K-fold splits:

| Split File | Temporal Scope | Approx. Rows | Split Ratio | Intended Usage |
| :--- | :--- | :--- | :--- | :--- |
| `data/splits/train.parquet` | Months 1 through 8 (Jan – Aug) | ~30.8 Million | ~67% | Feature scaling, hyperparameter search, baseline training |
| `data/splits/val.parquet` | Months 9 and 10 (Sep – Oct) | ~7.7 Million | ~17% | Early stopping, model selection, threshold calibration |
| `data/splits/test.parquet` | Months 11 and 12 (Nov – Dec) | ~7.5 Million | ~16% | Final out-of-time evaluation & Datathon benchmark metrics |

---

### 7. How to Load and Train in Subsequent Notebooks

Because the split files contain millions of rows, load them efficiently using PyArrow or sample when prototyping:

```python
import pyarrow.parquet as pq
import pandas as pd

# Option 1: Fast Subsampled Prototype Training (Recommended for local experiments)
table = pq.read_table("data/splits/train.parquet", columns=PRE_TRIP_FEATURES + ["base_fare", "trip_duration_minutes"])
train_df = table.slice(0, 500_000).to_pandas()

# Option 2: Full Split Loading (For GPU / high-RAM training instances)
# train_df = pd.read_parquet("data/splits/train.parquet")
# val_df   = pd.read_parquet("data/splits/val.parquet")
# test_df  = pd.read_parquet("data/splits/test.parquet")
```

