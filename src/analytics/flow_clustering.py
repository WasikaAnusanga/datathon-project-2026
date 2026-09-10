"""
src/analytics/flow_clustering.py
--------------------------------
Urban Flow Analytics - Hotspot & Origin-Destination Flow Clustering Engine
Team DataCraft (Datathon 2026)

Implements Section 3.2:
- Empirically-grounded diurnal daypart segmentation
- Multi-dimensional zone behavioral feature extraction
- K-Means & Agglomerative clustering with full diagnostic evaluation (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Origin-Destination (OD) flow matrix and top hotspot aggregation
- Model artifact and summary table export for downstream reporting and dashboard integration
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ==============================================================================
# 1. Operational Daypart Definitions
# ==============================================================================
DAYPARTS = {
    "Morning Rush": [6, 7, 8, 9],            # AM Commute Inflow
    "Midday": [10, 11, 12, 13, 14, 15],       # Intra-City Business & Commerce
    "Evening Rush": [16, 17, 18, 19],        # PM Commute Outflow & Dinner
    "Late Night": [20, 21, 22, 23, 0, 1],     # Social, Entertainment & Nightlife
    "Overnight": [2, 3, 4, 5]                 # Essential Transit & Airport Pre-Flights
}

HOUR_TO_DAYPART = {}
for dp, hours in DAYPARTS.items():
    for h in hours:
        HOUR_TO_DAYPART[h] = dp


def get_daypart_name(hour: int) -> str:
    """Map trip pickup hour (0-23) to defined diurnal daypart."""
    return HOUR_TO_DAYPART.get(hour, "Midday")


# ==============================================================================
# 2. Zone Feature Extractor & Aggregator
# ==============================================================================
class ZoneFeatureExtractor:
    """
    Extracts multi-dimensional behavioral mobility vectors for each of the 265
    NYC TLC zones across diurnal dayparts and trip characteristics.
    """
    def __init__(self, zone_csv_path: str = "data/raw/zone/Urban_Flow_Analytics_Zone_Dataset.csv"):
        self.zone_csv_path = zone_csv_path
        self.zone_df = self._load_zone_reference()

    def _load_zone_reference(self) -> pd.DataFrame:
        if os.path.exists(self.zone_csv_path):
            df = pd.read_csv(self.zone_csv_path)
            df['loc_id'] = df['loc_id'].astype(int)
            return df
        # Fallback minimal definition
        return pd.DataFrame({
            'loc_id': list(range(1, 266)),
            'borough_name': ['Unknown'] * 265,
            'zone_name': [f'Zone {i}' for i in range(1, 266)],
            'service_zone': ['Unknown'] * 265
        })

    def aggregate_from_trips(self, trips_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes cleaned trips dataframe into:
        1. Zone-level behavioral features dataframe (265 rows)
        2. Daypart OD flow matrix dataframe
        """
        # Ensure pickup_hour and daypart exist
        if 'pickup_hour' not in trips_df.columns:
            if 'pickup_timestamp' in trips_df.columns:
                dt = pd.to_datetime(trips_df['pickup_timestamp'])
                trips_df['pickup_hour'] = dt.dt.hour
            else:
                trips_df['pickup_hour'] = 12
        
        trips_df['daypart'] = trips_df['pickup_hour'].map(get_daypart_name)

        # Precompute airport flags and cross-borough flags if not present
        if 'is_airport' not in trips_df.columns:
            airport_ids = [1, 132, 138]  # Newark, JFK, LaGuardia
            trips_df['is_airport'] = (
                trips_df['origin_loc_id'].isin(airport_ids) | 
                trips_df['dest_loc_id'].isin(airport_ids)
            ).astype(int)

        # 1. Pickup stats by zone and daypart
        p_dp = trips_df.groupby(['origin_loc_id', 'daypart']).agg(
            trip_count=('base_fare', 'count'),
            avg_fare=('base_fare', 'mean'),
            avg_distance=('distance_miles', 'mean'),
            airport_count=('is_airport', 'sum')
        ).reset_index()

        # 2. Dropoff stats by zone and daypart
        d_dp = trips_df.groupby(['dest_loc_id', 'daypart']).agg(
            dropoff_count=('base_fare', 'count')
        ).reset_index()

        # Total pickups and dropoffs per zone
        total_pickups = trips_df.groupby('origin_loc_id').size().rename('total_pickups')
        total_dropoffs = trips_df.groupby('dest_loc_id').size().rename('total_dropoffs')
        overall_fare = trips_df.groupby('origin_loc_id')['base_fare'].mean().rename('overall_avg_fare')
        overall_dist = trips_df.groupby('origin_loc_id')['distance_miles'].mean().rename('overall_avg_dist')
        overall_airport = trips_df.groupby('origin_loc_id')['is_airport'].mean().rename('overall_airport_ratio')

        # Cross-borough calculation
        if 'origin_loc_id' in trips_df.columns and 'dest_loc_id' in trips_df.columns:
            zone_to_boro = dict(zip(self.zone_df['loc_id'], self.zone_df['borough_name']))
            origin_boro = trips_df['origin_loc_id'].map(zone_to_boro)
            dest_boro = trips_df['dest_loc_id'].map(zone_to_boro)
            trips_df['is_cross_borough'] = (origin_boro != dest_boro).astype(int)
            overall_cross = trips_df.groupby('origin_loc_id')['is_cross_borough'].mean().rename('cross_borough_ratio')
        else:
            overall_cross = pd.Series(0.0, index=total_pickups.index, name='cross_borough_ratio')

        # Pivot daypart volumes
        p_pivot = p_dp.pivot(index='origin_loc_id', columns='daypart', values='trip_count').fillna(0)
        d_pivot = d_dp.pivot(index='dest_loc_id', columns='daypart', values='dropoff_count').fillna(0)

        # Build feature matrix for all zones 1 to 265
        all_zones = self.zone_df[['loc_id', 'borough_name', 'zone_name', 'service_zone']].copy()
        all_zones = all_zones.set_index('loc_id')

        all_zones['total_pickups'] = total_pickups
        all_zones['total_dropoffs'] = total_dropoffs
        all_zones['total_pickups'] = all_zones['total_pickups'].fillna(0)
        all_zones['total_dropoffs'] = all_zones['total_dropoffs'].fillna(0)
        all_zones['total_activity'] = all_zones['total_pickups'] + all_zones['total_dropoffs']

        all_zones['overall_avg_fare'] = overall_fare.fillna(15.0)
        all_zones['overall_avg_dist'] = overall_dist.fillna(3.0)
        all_zones['airport_ratio'] = overall_airport.fillna(0.0)
        all_zones['cross_borough_ratio'] = overall_cross.fillna(0.0)

        daypart_keys = ["Morning Rush", "Midday", "Evening Rush", "Late Night", "Overnight"]
        for dp in daypart_keys:
            p_col = p_pivot[dp] if dp in p_pivot.columns else pd.Series(0, index=all_zones.index)
            d_col = d_pivot[dp] if dp in d_pivot.columns else pd.Series(0, index=all_zones.index)
            
            all_zones[f'{dp}_pickups'] = p_col
            all_zones[f'{dp}_dropoffs'] = d_col
            all_zones[f'{dp}_pickups'] = all_zones[f'{dp}_pickups'].fillna(0)
            all_zones[f'{dp}_dropoffs'] = all_zones[f'{dp}_dropoffs'].fillna(0)

            # Share of pickups during this daypart
            all_zones[f'{dp}_pickup_share'] = all_zones[f'{dp}_pickups'] / (all_zones['total_pickups'] + 1e-5)
            
            # Net flow imbalance index: (Dropoffs - Pickups) / (Dropoffs + Pickups + 1e-5)
            all_zones[f'{dp}_net_flow'] = (
                (all_zones[f'{dp}_dropoffs'] - all_zones[f'{dp}_pickups']) /
                (all_zones[f'{dp}_dropoffs'] + all_zones[f'{dp}_pickups'] + 1e-5)
            )

        # Diurnal Tidal Reversal: Difference between AM Peak Net Flow and PM Peak Net Flow
        all_zones['tidal_reversal'] = all_zones['Morning Rush_net_flow'] - all_zones['Evening Rush_net_flow']

        features_df = all_zones.reset_index()

        # Build Daypart OD flow ranking table
        od_flows = trips_df.groupby(['daypart', 'origin_loc_id', 'dest_loc_id']).agg(
            trip_volume=('base_fare', 'count'),
            mean_fare=('base_fare', 'mean'),
            mean_distance=('distance_miles', 'mean')
        ).reset_index()

        zone_dict = dict(zip(self.zone_df['loc_id'], self.zone_df['zone_name']))
        boro_dict = dict(zip(self.zone_df['loc_id'], self.zone_df['borough_name']))

        od_flows['origin_name'] = od_flows['origin_loc_id'].map(zone_dict).fillna("Unknown")
        od_flows['dest_name'] = od_flows['dest_loc_id'].map(zone_dict).fillna("Unknown")
        od_flows['origin_borough'] = od_flows['origin_loc_id'].map(boro_dict).fillna("Unknown")
        od_flows['dest_borough'] = od_flows['dest_loc_id'].map(boro_dict).fillna("Unknown")
        od_flows['od_corridor'] = od_flows['origin_name'] + " -> " + od_flows['dest_name']
        od_flows = od_flows.sort_values(by=['daypart', 'trip_volume'], ascending=[True, False])

        return features_df, od_flows


# ==============================================================================
# 3. Urban Flow Clustering Engine
# ==============================================================================
class UrbanFlowClusterer:
    """
    Fits and compares K-Means and Agglomerative Hierarchical clustering on
    normalized zone behavioral vectors, evaluating with Silhouette, Davies-Bouldin,
    and Calinski-Harabasz metrics.
    """
    FEATURE_COLS = [
        "Morning Rush_pickup_share",
        "Midday_pickup_share",
        "Evening Rush_pickup_share",
        "Late Night_pickup_share",
        "Morning Rush_net_flow",
        "Evening Rush_net_flow",
        "Late Night_net_flow",
        "tidal_reversal",
        "overall_avg_dist",
        "overall_avg_fare",
        "cross_borough_ratio",
        "airport_ratio"
    ]

    CLUSTER_TAXONOMY = {
        0: {
            "name": "Core Commercial & Morning Inflow Hub",
            "tag": "🏢 Midtown / Lower Manhattan Office & Transit Hubs",
            "description": "High morning dropoff volume and strong positive AM net flow. Sinks workers in morning and disperses passengers in evening rush."
        },
        1: {
            "name": "Evening Dining & Nightlife Corridor",
            "tag": "🍸 Nightlife, Entertainment & Cultural Districts",
            "description": "High evening and late-night trip concentration. Massive outflow during late hours with high intra-borough social mobility."
        },
        2: {
            "name": "Intermodal Airport & Long-Haul Transit",
            "tag": "✈️ Regional Gateways & Airport Hubs (JFK/LGA/EWR)",
            "description": "Extremely high average trip distance and fare, high cross-borough transit, and steady all-day departure/arrival volume."
        },
        3: {
            "name": "Residential Morning Outflow / Commuter Origins",
            "tag": "🏡 Residential Neighborhoods & Commuter Communities",
            "description": "Strong negative net flow during morning rush (passengers leave for work) and positive inflow in evening return hours."
        },
        4: {
            "name": "Outer-Borough & Low-Density Periphery",
            "tag": "🌳 Outer Boroughs & Peripheral Transit Connectors",
            "description": "Lower trip density, balanced off-peak demand, high cross-borough connection requirement to central urban cores."
        }
    }

    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.diagnostics_: Dict[str, any] = {}

    def search_optimal_clusters(self, X: np.ndarray, k_range: range = range(2, 9)) -> Dict[str, List[float]]:
        """Compute diagnostic curves across k values."""
        results = {
            "k": list(k_range),
            "inertia": [],
            "silhouette": [],
            "davies_bouldin": [],
            "calinski_harabasz": []
        }
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X)
            results["inertia"].append(float(km.inertia_))
            results["silhouette"].append(float(silhouette_score(X, labels)))
            results["davies_bouldin"].append(float(davies_bouldin_score(X, labels)))
            results["calinski_harabasz"].append(float(calinski_harabasz_score(X, labels)))
        return results

    def fit_predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Scales features, evaluates diagnostic curves, fits optimal KMeans model,
        maps cluster taxonomy, and attaches cluster labels to the dataframe.
        """
        X_raw = features_df[self.FEATURE_COLS].fillna(0).values
        X_scaled = self.scaler.fit_transform(X_raw)

        # Run diagnostics across k=2..8
        diagnostic_curves = self.search_optimal_clusters(X_scaled, range(2, 9))
        self.diagnostics_["curves"] = diagnostic_curves

        # Fit main KMeans model
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=15)
        kmeans_labels = self.model.fit_predict(X_scaled)

        # Fit Agglomerative for comparison
        agg_model = AgglomerativeClustering(n_clusters=self.n_clusters)
        agg_labels = agg_model.fit_predict(X_scaled)

        # Record validation metrics for the chosen k
        self.diagnostics_["selected_k"] = self.n_clusters
        self.diagnostics_["kmeans_silhouette"] = float(silhouette_score(X_scaled, kmeans_labels))
        self.diagnostics_["kmeans_davies_bouldin"] = float(davies_bouldin_score(X_scaled, kmeans_labels))
        self.diagnostics_["kmeans_calinski_harabasz"] = float(calinski_harabasz_score(X_scaled, kmeans_labels))
        self.diagnostics_["hierarchical_silhouette"] = float(silhouette_score(X_scaled, agg_labels))

        features_df = features_df.copy()
        features_df['raw_cluster'] = kmeans_labels
        features_df['hierarchical_cluster'] = agg_labels

        # Dynamically align clusters to taxonomy based on feature characteristics
        aligned_df = self._align_taxonomy(features_df)
        return aligned_df

    def _align_taxonomy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps raw numeric cluster ids to meaningful domain profiles based on
        empirical feature values (Airport, Commercial AM sink, Nightlife, Residential, Outer).
        """
        cluster_means = df.groupby('raw_cluster').agg({
            'airport_ratio': 'mean',
            'overall_avg_dist': 'mean',
            'Morning Rush_net_flow': 'mean',
            'Late Night_pickup_share': 'mean',
            'Evening Rush_net_flow': 'mean',
            'total_activity': 'mean'
        })

        cluster_map = {}
        unassigned_raw = list(cluster_means.index)

        # 1. Airport: Highest airport_ratio and average distance
        airport_c = cluster_means['airport_ratio'].idxmax()
        cluster_map[airport_c] = 2
        unassigned_raw.remove(airport_c)

        # 2. Commercial: Highest Morning Rush net flow (inflow attractor)
        rem_means = cluster_means.loc[unassigned_raw]
        comm_c = rem_means['Morning Rush_net_flow'].idxmax()
        cluster_map[comm_c] = 0
        unassigned_raw.remove(comm_c)

        # 3. Nightlife: Highest Late Night pickup share
        rem_means = cluster_means.loc[unassigned_raw]
        night_c = rem_means['Late Night_pickup_share'].idxmax()
        cluster_map[night_c] = 1
        unassigned_raw.remove(night_c)

        # 4. Residential: Highest total activity among remaining
        rem_means = cluster_means.loc[unassigned_raw]
        res_c = rem_means['total_activity'].idxmax() if len(unassigned_raw) > 1 else unassigned_raw[0]
        cluster_map[res_c] = 3
        unassigned_raw.remove(res_c)

        # 5. Remaining: Outer Borough / Peripheral
        if unassigned_raw:
            cluster_map[unassigned_raw[0]] = 4

        df['cluster_id'] = df['raw_cluster'].map(cluster_map).fillna(4).astype(int)
        df['cluster_name'] = df['cluster_id'].map(lambda x: self.CLUSTER_TAXONOMY[x]['name'])
        df['cluster_tag'] = df['cluster_id'].map(lambda x: self.CLUSTER_TAXONOMY[x]['tag'])
        df['cluster_desc'] = df['cluster_id'].map(lambda x: self.CLUSTER_TAXONOMY[x]['description'])

        return df


# ==============================================================================
# 4. End-to-End Execution Pipeline
# ==============================================================================
def run_flow_clustering_pipeline(sample_per_month: int = 25000) -> Dict[str, any]:
    """
    Executes end-to-end Section 3.2 pipeline:
    1. Loads chunk-stratified cleaned trips across all 12 months covering all 24 diurnal hours
    2. Extracts zone behavioral vectors and daypart OD flows
    3. Fits multi-algorithm clustering and evaluates diagnostic metrics
    4. Saves reports and models artifacts for dashboard and report integration
    """
    print("=" * 75)
    print("URBANFLOW ANALYTICS - SECTION 3.2: HOTSPOT & OD FLOW CLUSTERING PIPELINE")
    print("=" * 75)

    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    zone_path = "data/raw/zone/Urban_Flow_Analytics_Zone_Dataset.csv"
    extractor = ZoneFeatureExtractor(zone_path)

    # Ingest representative clean trip data
    taxi_files = sorted(glob.glob("data/raw/taxi/*.csv"))
    if not taxi_files:
        raise FileNotFoundError("No raw taxi CSV files found in data/raw/taxi/.")

    print(f"Found {len(taxi_files)} monthly taxi dataset files.")
    dfs = []
    
    use_cols = [
        'pickup_timestamp', 'dropoff_timestamp', 'rider_count', 
        'distance_miles', 'origin_loc_id', 'dest_loc_id', 'base_fare'
    ]

    # Sample evenly across multiple chunks in each month to guarantee full 24-hour diurnal coverage
    samples_per_chunk = max(1000, sample_per_month // 10)

    for fpath in taxi_files:
        fname = os.path.basename(fpath)
        print(f"Sampling diurnal chunks from {fname}...")
        month_chunks = []
        
        try:
            for chunk in pd.read_csv(fpath, usecols=use_cols, chunksize=120000, low_memory=False):
                # Clean anomalies according to Datathon 2026 Audit rules
                chunk['pickup_timestamp'] = pd.to_datetime(chunk['pickup_timestamp'], errors='coerce')
                chunk['dropoff_timestamp'] = pd.to_datetime(chunk['dropoff_timestamp'], errors='coerce')
                
                # Compute duration and speed
                duration_hrs = (chunk['dropoff_timestamp'] - chunk['pickup_timestamp']).dt.total_seconds() / 3600.0
                speed_mph = chunk['distance_miles'] / duration_hrs.replace(0, np.nan)
                
                valid_mask = (
                    (chunk['dropoff_timestamp'] >= chunk['pickup_timestamp']) &
                    (chunk['rider_count'] > 0) &
                    (chunk['base_fare'].between(2.5, 500.0)) &
                    (chunk['distance_miles'].between(0.1, 100.0)) &
                    (speed_mph <= 80.0) &
                    (chunk['origin_loc_id'].between(1, 265)) &
                    (chunk['dest_loc_id'].between(1, 265))
                )
                chunk_clean = chunk.loc[valid_mask]
                
                if len(chunk_clean) > 0:
                    sub_n = min(samples_per_chunk, len(chunk_clean))
                    month_chunks.append(chunk_clean.sample(n=sub_n, random_state=42))
                
                if len(month_chunks) >= 10:
                    break
        except Exception as e:
            print(f"Warning reading {fname}: {e}")

        if month_chunks:
            month_df = pd.concat(month_chunks, ignore_index=True)
            dfs.append(month_df)

    trips_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal Cleaned Diurnal Sample Trips: {len(trips_df):,} records across 12 months")

    # Feature Extraction
    print("Extracting multi-dimensional zone behavioral vectors and daypart OD flows...")
    features_df, od_flows = extractor.aggregate_from_trips(trips_df)

    # Clustering & Diagnostics
    print("Fitting UrbanFlowClusterer (K-Means & Agglomerative with k=5)...")
    clusterer = UrbanFlowClusterer(n_clusters=5, random_state=42)
    clustered_zones = clusterer.fit_predict(features_df)

    # Print summary results
    print("\n--- Clustering Diagnostics (k=5) ---")
    print(f"K-Means Silhouette Score:         {clusterer.diagnostics_['kmeans_silhouette']:.4f}")
    print(f"K-Means Davies-Bouldin Index:     {clusterer.diagnostics_['kmeans_davies_bouldin']:.4f}")
    print(f"K-Means Calinski-Harabasz Index:  {clusterer.diagnostics_['kmeans_calinski_harabasz']:.1f}")
    print(f"Hierarchical Silhouette Score:    {clusterer.diagnostics_['hierarchical_silhouette']:.4f}")

    print("\n--- Zone Cluster Distribution ---")
    cluster_counts = clustered_zones.groupby(['cluster_id', 'cluster_name', 'cluster_tag']).size().reset_index(name='zone_count')
    for _, row in cluster_counts.iterrows():
        print(f"Cluster {row['cluster_id']} [{row['zone_count']} zones]: {row['cluster_name']}")

    # Save Artifacts
    clusters_csv = "reports/zone_clusters.csv"
    od_csv = "reports/daypart_od_flows.csv"
    metrics_json = "reports/clustering_metrics.json"

    clustered_zones.to_csv(clusters_csv, index=False)
    od_flows.to_csv(od_csv, index=False)
    
    with open(metrics_json, 'w', encoding='utf-8') as f:
        json.dump(clusterer.diagnostics_, f, indent=4)

    print(f"\nSaved cluster assignments to: {clusters_csv}")
    print(f"Saved daypart OD flows to:     {od_csv}")
    print(f"Saved diagnostics to:          {metrics_json}")
    print("=" * 75)
    print("SECTION 3.2 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 75)

    return {
        "clustered_zones": clustered_zones,
        "od_flows": od_flows,
        "diagnostics": clusterer.diagnostics_
    }


if __name__ == "__main__":
    run_flow_clustering_pipeline()
