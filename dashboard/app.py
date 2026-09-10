"""
Urban Flow Analytics - Streamlit Interactive Dashboard
Team: DataCraft (Datathon 2026)
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

st.set_page_config(
    page_title="UrbanFlow Analytics | DataCraft",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_zone_data():
    zone_path = "data/raw/zone/Urban_Flow_Analytics_Zone_Dataset.csv"
    if os.path.exists(zone_path):
        df = pd.read_csv(zone_path)
        return df
    return pd.DataFrame({
        'loc_id': [138, 230, 161, 236],
        'zone_name': ['LaGuardia Airport', 'Times Sq/Theatre District', 'Midtown Center', 'Upper East Side North'],
        'borough_name': ['Queens', 'Manhattan', 'Manhattan', 'Manhattan'],
        'service_zone': ['Airports', 'Yellow Zone', 'Yellow Zone', 'Yellow Zone']
    })

@st.cache_resource
def load_fare_pipeline():
    pkl_path = "models/fare_prediction_pipeline.pkl"
    joblib_path = "models/fare_prediction_pipeline.joblib"
    
    target_path = pkl_path if os.path.exists(pkl_path) else joblib_path
    if os.path.exists(target_path):
        from src.models.fare_model import UpfrontFarePipeline
        try:
            return UpfrontFarePipeline.load(target_path)
        except Exception as e:
            st.error(f"Error loading pipeline from {target_path}: {e}")
            return None
    return None

def render_fare_prediction_tab(zone_df, pipeline):
    st.header("🚕 No-Surprises Upfront Pricing Calculator")
    st.caption("Section 2.1 — Base Fare Prediction before trip start (Zero Post-Trip Leakage)")
    
    col1, col2 = st.columns([1, 1])
    
    zone_options = {
        int(row['loc_id']): f"{row['zone_name']} ({row['borough_name']})"
        for _, row in zone_df.iterrows()
    }
    zone_ids = list(zone_options.keys())
    
    with col1:
        st.subheader("📍 Trip Route & Locations")
        origin_id = st.selectbox(
            "Pickup Zone (Origin)",
            options=zone_ids,
            format_func=lambda x: zone_options[x],
            index=min(137, len(zone_ids)-1)
        )
        dest_id = st.selectbox(
            "Dropoff Zone (Destination)",
            options=zone_ids,
            format_func=lambda x: zone_options[x],
            index=min(229, len(zone_ids)-1)
        )
        
        origin_info = zone_df[zone_df['loc_id'] == origin_id].iloc[0]
        dest_info = zone_df[zone_df['loc_id'] == dest_id].iloc[0]
        
        st.info(f"**Route Overview**: {origin_info['borough_name']} ({origin_info['zone_name']}) ➔ {dest_info['borough_name']} ({dest_info['zone_name']})")
        
        st.subheader("⚙️ Booking Parameters")
        distance_miles = st.number_input("Estimated Trip Distance (Miles)", min_value=0.1, max_value=100.0, value=4.5, step=0.1)
        rider_count = st.slider("Passenger Count", min_value=1, max_value=6, value=1)
        rate_class = st.selectbox("Rate Class Tariff", options=[1, 2, 3, 4, 5, 6],
                                  format_func=lambda x: {1: "1 - Standard Rate", 2: "2 - JFK Airport Flat Rate", 3: "3 - Newark Airport Rate", 4: "4 - Westchester/Nassau", 5: "5 - Negotiated Fare", 6: "6 - Group Ride"}[x])
        provider_code = st.radio("Dispatch Provider Code", options=[1, 2], horizontal=True)

    with col2:
        st.subheader("📅 Schedule & Timing")
        pickup_date = st.date_input("Pickup Date", datetime.date.today())
        pickup_time = st.time_input("Pickup Time", datetime.time(17, 30))
        
        pickup_dt = datetime.datetime.combine(pickup_date, pickup_time)
        
        st.write("---")
        st.subheader("💵 Upfront Base Fare Estimation")
        
        trip_payload = {
            'provider_code': provider_code,
            'rider_count': rider_count,
            'distance_miles': distance_miles,
            'rate_class_id': rate_class,
            'origin_loc_id': origin_id,
            'dest_loc_id': dest_id,
            'pickup_timestamp': pickup_dt.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if pipeline is not None:
            est_fare = pipeline.predict_single(trip_payload)
            st.metric(label="Predicted Base Fare", value=f"${est_fare:.2f}")
            st.success("✅ Prediction generated using pre-trip features with zero post-trip leakage!")
        else:
            base_rate = 3.00 + (distance_miles * 2.75)
            if rate_class == 2:
                base_rate = 70.00
            st.metric(label="Estimated Base Fare (Rule Baseline)", value=f"${base_rate:.2f}")
            st.warning("⚠️ Model pipeline artifact loading... Showing baseline heuristic estimate.")

        is_cross = origin_info['borough_name'] != dest_info['borough_name']
        is_airport = (rate_class in [2, 3]) or ("Airport" in origin_info['zone_name']) or ("Airport" in dest_info['zone_name'])
        
        with st.expander("🔍 Pre-Trip Feature Breakdown", expanded=True):
            st.write(f"- **Cross-Borough Trip**: {'Yes 🌉' if is_cross else 'No 🏙️'}")
            st.write(f"- **Airport Trip**: {'Yes ✈️' if is_airport else 'No 🚗'}")
            st.write(f"- **Pickup Hour**: {pickup_dt.hour}:00 ({'Peak Rush Hour 🚦' if (pickup_dt.weekday() < 5 and (7 <= pickup_dt.hour <= 10 or 16 <= pickup_dt.hour <= 20)) else 'Off-Peak 🟢'})")
            st.write(f"- **Day of Week**: {pickup_dt.strftime('%A')}")

@st.cache_data
def load_clustering_data():
    clusters_path = "reports/zone_clusters.csv"
    od_path = "reports/daypart_od_flows.csv"
    metrics_path = "reports/clustering_metrics.json"

    if not (os.path.exists(clusters_path) and os.path.exists(od_path)):
        from src.analytics.flow_clustering import run_flow_clustering_pipeline
        run_flow_clustering_pipeline()

    clusters_df = pd.read_csv(clusters_path)
    od_df = pd.read_csv(od_path)

    metrics = {}
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

    return clusters_df, od_df, metrics

def render_clustering_tab(zone_df, clusters_df, od_df, metrics):
    import plotly.express as px
    import plotly.graph_objects as go

    st.header("🗺️ Hotspot & Origin-Destination Flow Clustering")
    st.caption("Section 3.2 — Macro-level Urban Mobility Archetypes & Diurnal Movement Dynamics (Morning Rush to Late Night)")

    # Top KPI Metrics Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Analyzed TLC Zones", f"{len(clusters_df)} Zones", help="Full coverage across Manhattan, Queens, Brooklyn, Bronx, and Staten Island")
    with kpi2:
        st.metric("Mobility Clusters", f"{clusters_df['cluster_id'].nunique()} Archetypes", help="Optimal cluster count selected via Silhouette & Elbow diagnostics")
    with kpi3:
        sil_score = metrics.get('kmeans_silhouette', 0.2819)
        st.metric("Silhouette Score", f"{sil_score:.4f}", help="K-Means cluster cohesion vs separation index")
    with kpi4:
        st.metric("Unique OD Corridors", f"{len(od_df):,}", help="Origin-Destination movement pairs tracked across diurnal dayparts")

    st.write("---")

    tab_flow, tab_clusters = st.tabs([
        "🕒 Diurnal Hotspots & OD Flow Corridors",
        "🏷️ Urban Mobility Cluster Taxonomy"
    ])

    # -------------------------------------------------------------------------
    # SUB-TAB 1: Hotspots & OD Flows by Daypart
    # -------------------------------------------------------------------------
    with tab_flow:
        st.subheader("Diurnal Movement & Corridor Shift Analysis")
        st.write("City planners can select a diurnal daypart to track how passenger origin and destination hotspots invert from morning commute hours to late-night entertainment hours.")

        col_filter1, col_filter2 = st.columns([1, 1])
        with col_filter1:
            daypart_options = ["Morning Rush", "Midday", "Evening Rush", "Late Night", "Overnight"]
            selected_daypart = st.selectbox("Select Operational Daypart", daypart_options, index=0)
        with col_filter2:
            borough_list = ["All Boroughs"] + sorted(zone_df['borough_name'].dropna().unique().tolist())
            selected_borough = st.selectbox("Filter by Origin Borough", borough_list, index=0)

        # Filter OD data
        sub_od = od_df[od_df['daypart'] == selected_daypart].copy()
        if selected_borough != "All Boroughs":
            sub_od = sub_od[sub_od['origin_borough'] == selected_borough]

        # Top Hotspots Visuals
        col_chart1, col_chart2 = st.columns(2)
        top_origins = sub_od.groupby('origin_name')['trip_volume'].sum().sort_values(ascending=False).head(10).reset_index()
        top_dests = sub_od.groupby('dest_name')['trip_volume'].sum().sort_values(ascending=False).head(10).reset_index()

        with col_chart1:
            st.markdown(f"**Top 10 Passenger Pickups (Sources) — {selected_daypart}**")
            if not top_origins.empty:
                fig_orig = px.bar(
                    top_origins.sort_values('trip_volume', ascending=True),
                    x='trip_volume', y='origin_name',
                    orientation='h',
                    labels={'trip_volume': 'Rider Pickups', 'origin_name': 'Pickup Zone'},
                    color='trip_volume',
                    color_continuous_scale='Blues'
                )
                fig_orig.update_layout(showlegend=False, height=360, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_orig, use_container_width=True)
            else:
                st.info("No trips found for selected borough filter.")

        with col_chart2:
            st.markdown(f"**Top 10 Passenger Dropoffs (Sinks) — {selected_daypart}**")
            if not top_dests.empty:
                fig_dest = px.bar(
                    top_dests.sort_values('trip_volume', ascending=True),
                    x='trip_volume', y='dest_name',
                    orientation='h',
                    labels={'trip_volume': 'Rider Dropoffs', 'dest_name': 'Destination Zone'},
                    color='trip_volume',
                    color_continuous_scale='Teal'
                )
                fig_dest.update_layout(showlegend=False, height=360, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_dest, use_container_width=True)
            else:
                st.info("No trips found for selected borough filter.")

        st.subheader(f"Top 15 Route Corridors in {selected_daypart}")
        display_od = sub_od.head(15)[['od_corridor', 'origin_borough', 'dest_borough', 'trip_volume', 'mean_fare', 'mean_distance']].copy()
        display_od.columns = ['Route Corridor', 'Origin Borough', 'Dest Borough', 'Trip Volume', 'Avg Base Fare ($)', 'Avg Distance (mi)']
        display_od['Avg Base Fare ($)'] = display_od['Avg Base Fare ($)'].map(lambda x: f"${x:.2f}")
        display_od['Avg Distance (mi)'] = display_od['Avg Distance (mi)'].map(lambda x: f"{x:.2f} mi")
        st.dataframe(display_od, use_container_width=True, hide_index=True)

        # Net Flow Tidal Imbalance Insight
        st.markdown("### ⚖️ Zone Net Inflow vs Outflow Imbalance Index")
        st.caption("Net Flow Index = (Dropoffs - Pickups) / Total Activity. Positive indicates a net attractor/sink; negative indicates an outflow source.")
        
        net_col = f"{selected_daypart}_net_flow"
        if net_col in clusters_df.columns:
            top_sinks = clusters_df.sort_values(by=net_col, ascending=False).head(5)[['zone_name', 'borough_name', net_col, 'total_activity']]
            top_sources = clusters_df.sort_values(by=net_col, ascending=True).head(5)[['zone_name', 'borough_name', net_col, 'total_activity']]
            
            c_sink, c_source = st.columns(2)
            with c_sink:
                st.success(f"**Top Net Attractors (Sinks) in {selected_daypart}:**\n" + 
                           "\n".join([f"- **{r['zone_name']}** ({r['borough_name']}): Net index `+{r[net_col]:.2f}`" for _, r in top_sinks.iterrows()]))
            with c_source:
                st.warning(f"**Top Net Generators (Sources) in {selected_daypart}:**\n" + 
                           "\n".join([f"- **{r['zone_name']}** ({r['borough_name']}): Net index `{r[net_col]:.2f}`" for _, r in top_sources.iterrows()]))

    # -------------------------------------------------------------------------
    # SUB-TAB 2: Urban Mobility Cluster Profiles & Taxonomy
    # -------------------------------------------------------------------------
    with tab_clusters:
        st.subheader("Urban Mobility Archetypes & Behavioral Profiles")
        st.write("Using 12 behavioral dimensions (temporal shares, net tidal flow ratios, airport links, and trip economics), K-Means groups NYC into 5 cohesive operational archetypes.")

        cluster_names = sorted(clusters_df['cluster_name'].unique().tolist())
        selected_cluster_name = st.selectbox("Select Cluster Archetype to Inspect", cluster_names, index=0)

        cluster_sub = clusters_df[clusters_df['cluster_name'] == selected_cluster_name]
        cluster_info = cluster_sub.iloc[0]

        st.info(f"**{cluster_info['cluster_tag']}**\n\n{cluster_info['cluster_desc']}")

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Total Zones in Cluster", f"{len(cluster_sub)} Zones")
        with mcol2:
            st.metric("Avg Trip Distance", f"{cluster_sub['overall_avg_dist'].mean():.2f} miles")
        with mcol3:
            st.metric("Avg Base Fare", f"${cluster_sub['overall_avg_fare'].mean():.2f}")
        with mcol4:
            st.metric("Late Night Trip Share", f"{cluster_sub['Late Night_pickup_share'].mean()*100:.1f}%")

        # Cluster Radar / Comparison
        st.markdown("#### Behavioral Profile Across Key Dimensions")
        radar_categories = [
            'Morning Rush Share', 'Midday Share', 'Evening Rush Share', 
            'Late Night Share', 'Cross-Borough Ratio', 'Airport Ratio'
        ]
        
        all_clusters_mean = clusters_df.groupby('cluster_name').agg({
            'Morning Rush_pickup_share': 'mean',
            'Midday_pickup_share': 'mean',
            'Evening Rush_pickup_share': 'mean',
            'Late Night_pickup_share': 'mean',
            'cross_borough_ratio': 'mean',
            'airport_ratio': 'mean'
        }).reset_index()

        fig_radar = go.Figure()
        for _, row in all_clusters_mean.iterrows():
            vals = [
                row['Morning Rush_pickup_share'], row['Midday_pickup_share'], 
                row['Evening Rush_pickup_share'], row['Late Night_pickup_share'],
                row['cross_borough_ratio'], row['airport_ratio']
            ]
            vals.append(vals[0])
            cats = radar_categories + [radar_categories[0]]
            is_selected = (row['cluster_name'] == selected_cluster_name)
            
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=cats,
                fill='toself' if is_selected else None,
                name=row['cluster_name'],
                line=dict(width=3 if is_selected else 1.5),
                opacity=0.9 if is_selected else 0.4
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 0.6])),
            showlegend=True,
            height=450,
            margin=dict(l=40, r=40, t=30, b=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("#### Member Zones Directory")
        st.dataframe(
            cluster_sub[['loc_id', 'borough_name', 'zone_name', 'service_zone', 'total_activity', 'overall_avg_fare', 'overall_avg_dist']].sort_values('total_activity', ascending=False),
            use_container_width=True,
            hide_index=True
        )

        with st.expander("🏛️ City Planning & Fleet Dispatch Policy Playbook", expanded=True):
            st.markdown("""
            - **Curb Space & Dedicated Taxi Staging:** In *Core Commercial & Morning Inflow Hubs*, convert curbside parking into designated 15-minute taxi passenger loading bays between 15:30 and 19:30 to mitigate traffic bottlenecks.
            - **Nightlife Fleet Repositioning:** Direct off-duty drivers to *Evening Dining & Nightlife Corridors* (East Village, Chelsea, SoHo) where post-22:00 demand spikes exceed midday levels.
            - **Airport Queue Coordination:** Implement virtual staging queues at *Intermodal Airport Hubs* matching arriving flights with passenger departures to eliminate empty deadhead trips across East River crossings.
            - **Outer-Borough Transit Feeder Integration:** In *Residential Morning Outflow Hubs*, provide subsidized flat-rate feeder trips to express subway terminals during the 06:00-09:00 AM commute peak.
            """)

def main():
    st.sidebar.title("🚕 DataCraft UrbanFlow")
    st.sidebar.markdown("**Datathon 2026**")
    
    navigation = st.sidebar.radio(
        "Platform Modules",
        [
            "🚕 Upfront Base Fare Estimator",
            "🗺️ Hotspot & OD Flow Clustering (Section 3.2)",
            "⏱️ ETA & Trip Duration (Teammate)",
            "📈 Demand Forecasting (Teammate)",
            "📊 Executive Overview & Quality"
        ]
    )
    
    zone_df = load_zone_data()
    pipeline = load_fare_pipeline()
    
    if navigation == "🚕 Upfront Base Fare Estimator":
        render_fare_prediction_tab(zone_df, pipeline)
    elif navigation == "🗺️ Hotspot & OD Flow Clustering (Section 3.2)":
        clusters_df, od_df, metrics = load_clustering_data()
        render_clustering_tab(zone_df, clusters_df, od_df, metrics)
    elif navigation == "⏱️ ETA & Trip Duration (Teammate)":
        st.header("⏱️ ETA & Trip Duration Modeling")
        st.info("Teammate component: Predicts expected trip duration (minutes) using pre-trip route features.")
    elif navigation == "📈 Demand Forecasting (Teammate)":
        st.header("📈 Urban Demand & Spatial Hotspots")
        st.info("Teammate component: Forecasts pickup volume and spatial demand heatmaps.")
    else:
        st.header("📊 Data Quality & System Overview")
        st.markdown("### Team DataCraft Pipeline Summary")
        st.write("- **Raw Records Analyzed**: 48,599,839 rows (12 months)")
        st.write("- **Clean Dataset Size**: 45,956,110 rows (94.56% retained)")
        st.write("- **Anomalies Dropped**: 2,643,729 rows (5.44%)")
        st.write("- **Section 2.1 Upfront Base Fare Model**: LightGBM Pipeline (Test MAE: $4.60, R²: 0.68)")
        st.write("- **Section 3.2 Hotspot & Flow Clustering**: 5 Diurnal Mobility Archetypes (Silhouette: 0.2819)")

if __name__ == "__main__":
    main()

