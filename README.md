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
│   ├── 03_fare_prediction.ipynb            # Section 2.1 Upfront Fare Prediction Notebook
│   └── 06_hotspot_od_analysis.ipynb        # Section 3.2 Hotspot & OD Flow Clustering Notebook
├── reports/
│   ├── zone_clusters.csv                   # Section 3.2 265 Zone Cluster Assignments
│   ├── daypart_od_flows.csv                # Section 3.2 Daypart OD Flow Corridors
│   ├── clustering_metrics.json             # Section 3.2 Clustering Validation Diagnostics
│   └── data_quality_summary.json           # Data Audit Summary Metrics
├── src/
│   ├── analytics/
│   │   └── flow_clustering.py              # Section 3.2 Clustering Engine & OD Flow Aggregator
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

### 4. Run Section 3.2 Hotspot & OD Flow Clustering Pipeline
To extract zone behavioral vectors, run multi-algorithm clustering diagnostics, and generate daypart flow tables:
```bash
python -m src.analytics.flow_clustering
```

### 5. Launch Interactive Streamlit Dashboard
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

---

## 🗺️ Section 3.2 — Hotspot & Origin-Destination Flow Clustering

### Business & City Planning Goal
City planners need macro-level spatial-temporal intelligence to manage urban congestion, mitigate empty vehicle cruising, and allocate curb infrastructure. Analyzing tens of millions of single taxi rides makes it impossible to see high-level mobility structure. By clustering pickup and drop-off patterns across key diurnal dayparts, this module maps out major travel hotspots, directional commuter reversals, and inter-borough movement corridors from morning work hours to late at night.

---

### 1. Diurnal Daypart Taxonomy
Urban transit demand follows distinct temporal rhythms. Rather than arbitrary clock divisions, we demarcate 5 operational dayparts grounded in empirical 24-hour volume distributions:

| Daypart | Time Window | Transit Characteristics & Operational Role |
| :--- | :---: | :--- |
| **Morning Rush (AM Peak)** | 06:00 – 10:00 | Inward commuter surge from residential hubs into commercial employment centers |
| **Midday / Commercial** | 10:00 – 16:00 | Intra-city business meetings, shopping, medical, and dining transit |
| **Evening Rush (PM Peak)** | 16:00 – 20:00 | Outward return commute from job centers towards residential and dinner destinations |
| **Late Night / Nightlife** | 20:00 – 02:00 | Entertainment, cultural nightlife, dining, and social gatherings |
| **Overnight** | 02:00 – 06:00 | Low-volume baseline, essential night-shift workers, and early airport shuttles |

---

### 2. Multi-Dimensional Zone Behavioral Vectors
For each of the 265 NYC TLC zones, we extract high-signal behavioral vectors:
- **Temporal Volume Shares:** Percentage of total trips occurring in each of the 5 diurnal dayparts.
- **Net Flow Imbalance Index ($I$):**  
  $$I = \frac{\text{Dropoffs} - \text{Pickups}}{\text{Dropoffs} + \text{Pickups} + 5}$$  
  Measures whether a zone acts as a **Net Attractor / Sink** ($I > 0$) or a **Net Generator / Source** ($I < 0$).
- **Diurnal Tidal Reversal:** Difference between Morning Rush net flow and Evening Rush net flow ($I_{\text{AM}} - I_{\text{PM}}$). Captures the classic commuting inversion where commercial nodes sink riders in the morning and disperse them in the evening.
- **Spatial & Economic Dimensions:** Average trip distance, average base fare, cross-borough trip ratio, and airport trip ratio.

---

### 3. Clustering Methodology & Empirical Diagnostics
Standardized feature vectors were clustered using **K-Means** across $k \in [2, 8]$ and cross-validated against **Agglomerative Hierarchical Clustering**:

| Metric | Score ($k=5$) | Interpretation |
| :--- | :---: | :--- |
| **Selected Cluster Count ($k$)** | **5** | Optimal trade-off between statistical cohesion and operational interpretability |
| **K-Means Silhouette Score** | **0.2819** | Strong cluster separation on high-dimensional spatial-temporal behavior |
| **Davies-Bouldin Index** | **1.2735** | Low intra-cluster dispersion relative to inter-cluster separation |
| **Calinski-Harabasz Index** | **73.2** | High between-cluster variance ratio |
| **Hierarchical Silhouette Score** | **0.2399** | Confirms stability across distinct clustering algorithm families |

---

### 4. Urban Mobility Archetypes (5 Clusters)

| Cluster ID & Archetype | Zone Count | Key Behavioral Attributes | Example NYC Zones |
| :--- | :---: | :--- | :--- |
| **Cluster 0: Core Commercial & Morning Inflow Hub** | 80 zones | High morning dropoffs, positive AM net flow ($+0.44$), strong midday business demand | East Harlem North, Two Bridges/Seward Park, Central Harlem North, Stuy Town |
| **Cluster 1: Evening Dining & Nightlife Corridor** | 59 zones | Highest trip volume density, massive late-night share ($24\%+$), short intra-borough rides | Midtown Center, Times Sq, Upper East Side, East Village, Chelsea, SoHo |
| **Cluster 2: Intermodal Airport & Long-Haul Transit** | 4 zones | Longest trips ($11.9$ mi), highest fares ($\$67.72$ avg), steady all-day arrivals/departures | JFK Airport, LaGuardia Airport, Newark Airport, Arrochar |
| **Cluster 3: Residential Morning Outflow / Commuter Origins** | 95 zones | Negative AM net flow ($-0.18$), morning departure source, evening inflow sink | Crown Heights North, East New York, Flatbush, Stuyvesant Heights, Bushwick |
| **Cluster 4: Outer-Borough & Low-Density Periphery** | 27 zones | Low yellow taxi density, short local transit or long cross-borough links, off-peak stability | Staten Island peripheral zones, Outer Queens |

---

### 5. Daypart OD Flow Dynamics (Morning Rush to Late Night)
- **Morning Rush Inversion:** Top corridors flow heavily from residential zones (Upper East/West, Brooklyn) inward to commercial centroids (Midtown Center, Grand Central, Financial District).
- **Evening Rush Dispersal:** Outward flow surges from Midtown/Lower Manhattan toward residential outer boroughs and Midtown East/West dining hubs.
- **Late-Night Entertainment Corridors:** Concentrated flow within Manhattan's cultural grid (East Village $\leftrightarrow$ Chelsea $\leftrightarrow$ Williamsburg $\leftrightarrow$ Lower East Side).

---

### 6. Submission & Reporting Artifacts
- **`notebooks/06_hotspot_od_analysis.ipynb`**: Fully executable, publication-grade Jupyter Notebook covering end-to-end analysis, diagnostic curves, cluster profiling, and city planning policy recommendations.
- **`src/analytics/flow_clustering.py`**: Reusable clustering engine, feature extraction pipeline, and daypart OD aggregator.
- **`reports/zone_clusters.csv`**: Complete 265-zone dataset with cluster IDs, taxonomy names, and behavioral attributes.
- **`reports/daypart_od_flows.csv`**: Ranked Origin-Destination movement corridors across all 5 dayparts.
- **`reports/clustering_metrics.json`**: Numerical validation metrics across $k=2 \dots 8$.
- **`dashboard/app.py`**: Interactive Streamlit dashboard with a dedicated **"🗺️ Hotspot & OD Flow Clustering"** tab featuring interactive daypart filtering, top OD flow matrices, and cluster radar profiles.

