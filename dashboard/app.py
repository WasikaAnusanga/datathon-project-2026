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

@st.cache_resource
def load_arrival_estimator():
    model_path = "models/arrival_time_estimator.pkl"
    if not os.path.exists(model_path):
        return None
    from src.models.arrival_model import ArrivalEstimator
    try:
        return ArrivalEstimator.load(model_path)
    except Exception as exc:
        st.error(f"Error loading arrival estimator: {exc}")
        return None

def get_query_planner():
    import importlib
    import src.assistant.llm_client
    import src.assistant.query_planner
    importlib.reload(src.assistant.llm_client)
    importlib.reload(src.assistant.query_planner)
    from src.assistant.query_planner import QueryPlanner
    return QueryPlanner(use_sample=False)

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

def render_arrival_prediction_tab(zone_df, estimator):
    st.header("⏱️ Pre-Trip ETA Estimator")
    st.caption("Predicts trip duration and arrival time before departure using zones, pickup timing, and training-only traffic history.")

    zone_options = {
        int(row["loc_id"]): f"{row['zone_name']} ({row['borough_name']})"
        for _, row in zone_df.iterrows()
    }
    zone_ids = list(zone_options)
    if not zone_ids:
        st.error("No taxi zones are available.")
        return

    col_route, col_time = st.columns(2)
    with col_route:
        st.subheader("📍 Route")
        origin_id = st.selectbox("Pickup zone", zone_ids, format_func=lambda value: zone_options[value], key="eta_origin_id")
        dest_id = st.selectbox("Destination zone", zone_ids, format_func=lambda value: zone_options[value], index=min(229, len(zone_ids) - 1), key="eta_dest_id")
        rider_count = st.slider("Passenger count", 1, 6, 1, key="eta_rider_count")
        provider_code = st.radio("Provider code", [1, 2], horizontal=True, key="eta_provider")
        rate_class = st.selectbox("Rate class", [1, 2, 3, 4, 5, 6], key="eta_rate_class")

    with col_time:
        st.subheader("📅 Departure")
        pickup_date = st.date_input("Pickup date", datetime.date.today(), key="eta_date")
        pickup_time = st.time_input("Pickup time", datetime.time(17, 30), key="eta_time")

    origin_info = zone_df[zone_df["loc_id"] == origin_id].iloc[0]
    dest_info = zone_df[zone_df["loc_id"] == dest_id].iloc[0]
    pickup_dt = datetime.datetime.combine(pickup_date, pickup_time)
    request = pd.DataFrame([{
        "pickup_timestamp": pickup_dt,
        "provider_code": provider_code,
        "rider_count": rider_count,
        "rate_class_id": rate_class,
        "origin_loc_id": origin_id,
        "dest_loc_id": dest_id,
        "origin_borough": origin_info["borough_name"],
        "dest_borough": dest_info["borough_name"],
        "origin_service_zone": origin_info["service_zone"],
        "dest_service_zone": dest_info["service_zone"],
        "origin_zone": origin_info["zone_name"],
        "dest_zone": dest_info["zone_name"],
    }])

    st.info(f"**Route:** {origin_info['zone_name']} ({origin_info['borough_name']}) → {dest_info['zone_name']} ({dest_info['borough_name']})")
    if estimator is None:
        st.warning("Arrival model artifact is not available. Train it with `src/models/train_arrival_pipeline.py`.")
        return

    predicted_minutes = float(estimator.predict_minutes(request)[0])
    arrival_time = estimator.predict_arrival(request).iloc[0]
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Predicted trip time", f"{predicted_minutes:.0f} min")
    metric_col2.metric("Estimated arrival", arrival_time.strftime("%H:%M"))
    metric_col3.metric("Model", "LightGBM ETA")
    st.success("ETA generated from pre-trip information only.")
    with st.expander("🔍 Prediction details", expanded=True):
        st.write(f"- **Pickup:** {pickup_dt.strftime('%Y-%m-%d %H:%M')}")
        st.write("- **Destination assumption:** entered before departure")
        st.write("- **Training policy:** trips up to 180 minutes; extreme records retained for audit")
        st.write("- **Features:** zones, pickup calendar, booking fields, and training-only historical OD traffic medians")

def render_ai_assistant_tab():
    st.header("🤖 AI Mobility Assistant (Track 5)")
    st.caption("Natural Language Text-to-SQL Analytics over 45.95M urban taxi records powered by DuckDB & SQLGlot")
    
    from src.assistant.response_builder import ResponseBuilder
    from src.assistant.chart_builder import ChartBuilder
    from src.assistant.audit_logger import AuditLogger
    
    planner = get_query_planner()
    logger = AuditLogger()

    # Session State management for text input
    if "user_query_text_box" not in st.session_state:
        st.session_state["user_query_text_box"] = ""

    st.markdown("#### 💡 Quick Sample Questions")
    col_q1, col_q2, col_q3, col_q4, col_q5 = st.columns(5)
    
    if col_q1.button("🏙️ Top Busiest Boroughs"):
        st.session_state["user_query_text_box"] = "What are the top 5 busiest pickup boroughs?"
    if col_q2.button("💵 Avg Fare Manhattan"):
        st.session_state["user_query_text_box"] = "What is the average trip fare in Manhattan?"
    if col_q3.button("⏱️ Hourly Speed Profile"):
        st.session_state["user_query_text_box"] = "Show me hourly speed distribution for taxi trips"
    if col_q4.button("❓ Ambiguous: Midtown"):
        st.session_state["user_query_text_box"] = "What is the average fare in Midtown?"
    if col_q5.button("⛔ Out-of-Scope: Weather"):
        st.session_state["user_query_text_box"] = "Join taxi trips with weather data"

    query_input = st.text_input(
        "Ask any natural language analytical question about NYC taxi trips:",
        placeholder="e.g. What are the top 5 busiest pickup boroughs?",
        key="user_query_text_box"
    )

    if hasattr(planner.llm_client, "check_api_key_status"):
        key_status = planner.llm_client.check_api_key_status()
        if not key_status.get("valid_format", True):
            st.info(f"ℹ️ **Engine Status**: {key_status['message']}")


    if query_input:
        with st.spinner("Processing question & compiling SQL..."):
            res = planner.process_question(query_input)

            # Audit log metadata
            logger.log_query_execution(
                question=query_input,
                intent_category=res.intent.intent_category.value if res.intent else "unknown",
                generated_sql=res.sql,
                validation_status="VALID" if res.success else "REJECTED",
                clarification_status=bool(res.clarification_prompt),
                execution_time_ms=res.meta.get("execution_time_ms", 0.0),
                returned_row_count=len(res.data),
                error_category=res.error_message if not res.success else None
            )

        if res.clarification_prompt:
            st.warning(f"❓ **Clarification Required**: {res.clarification_prompt}")
        elif not res.success:
            st.error(f"⛔ **Query Rejected / Out of Scope**: {res.error_message}")
        else:
            st.success("✅ Query compiled & executed safely!")
            
            # Grounded response answer
            grounded_answer = ResponseBuilder.build_response(query_input, res.data, res.meta)
            st.markdown(f"### 💬 Analytical Answer\n{grounded_answer}")

            # SQL expander view
            with st.expander("🔍 View Deterministic SQL & Execution Metrics", expanded=True):
                st.code(res.sql, language="sql")
                st.write(f"- **Execution Time**: `{res.meta.get('execution_time_ms', 0):.2f} ms` over 45,956,110 rows")
                st.write(f"- **Returned Rows**: `{len(res.data)}` (capped at {planner.executor.HARD_MAX_RESULT_ROWS})")

            # Visual plot
            fig = ChartBuilder.generate_chart(res.data, query_input)
            if fig is not None:
                st.subheader("📊 Visualization")
                st.plotly_chart(fig, use_container_width=True)

            # Data Table preview
            st.subheader("📋 Query Results Table")
            st.dataframe(res.data, use_container_width=True)

    planner.executor.close()

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
            """)

@st.cache_data
def load_business_data():
    pay_path = "reports/payment_tip_summary.csv"
    velocity_path = "reports/revenue_velocity.csv"
    deadhead_path = "reports/deadhead_corridors.csv"
    kpi_path = "reports/business_kpis.json"

    if not (os.path.exists(pay_path) and os.path.exists(velocity_path)):
        from src.analytics.business_analytics import run_business_analytics_pipeline
        run_business_analytics_pipeline()

    pay_df = pd.read_csv(pay_path)
    velocity_df = pd.read_csv(velocity_path)
    deadhead_df = pd.read_csv(deadhead_path)

    kpis = {}
    if os.path.exists(kpi_path):
        import json
        with open(kpi_path, 'r', encoding='utf-8') as f:
            kpis = json.load(f)

    return pay_df, velocity_df, deadhead_df, kpis

def render_business_decision_tab(pay_df, velocity_df, deadhead_df, kpis):
    import plotly.express as px
    import plotly.graph_objects as go
    from src.analytics.business_analytics import BusinessDecisionEngine

    st.header("💼 Executive Business Intelligence & Decision Engine")
    st.caption("Track 6 — Turning Taxi Telematics, Payment Behavior & Revenue Velocity into Strategic Value")

    # The 4-Phase Executive Story Banner
    with st.expander("📖 Executive Data-Driven Story: The Fleet Revenue Optimization Framework", expanded=True):
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.error("**1. What is the Problem?**\n- Driver hourly earnings compress by 35% in daytime gridlock.\n- 10.4% cash trips record zero digital tips.\n- Outer-borough dropoffs suffer up to 99.9% empty return deadheading.")
        with col_s2:
            st.info("**2. What Does the Data Say?**\n- Queens expressways yield $127/hr vs Manhattan midday at $82/hr.\n- Cards yield $4.30 tip (26.5%) vs Cash $0.00.\n- Surcharges & taxes take 13.2% of passenger charge.")
        with col_s3:
            st.warning("**3. Why is it Happening?**\n- Stop-and-go congestion drops speed to <8 mph.\n- Terminal UI lacks smart tip presets.\n- Outbound commuter flows lack return demand matching.")
        with col_s4:
            st.success("**4. What Should We Do?**\n- Smart In-Cab POS Tipping Presets (+2.5% tip lift).\n- Dynamic Outer-Borough Return Incentives.\n- Reposition drivers to high-velocity corridors.")

    # Top Executive KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Peak Revenue Velocity", "$127.81 / hr", delta="Queens Arterials", help="Highest gross driver hourly earning rate across NYC")
    with kpi2:
        st.metric("Congestion Drag Rate", "$82.83 / hr", delta="-35.2% vs Peak", delta_color="inverse", help="Depressed driver earning rate in Manhattan midday gridlock")
    with kpi3:
        st.metric("Digital Tip Capture Rate", "93.75%", delta="26.5% on Base Fare", help="Percentage of credit card rides leaving a digital tip")
    with kpi4:
        st.metric("Non-Fare Surcharge Share", "13.23%", delta="1 in 7 Dollars", delta_color="inverse", help="Share of passenger spend absorbed by taxes, congestion, and airport fees")

    st.write("---")

    tab_roi, tab_velocity, tab_payment, tab_deadhead = st.tabs([
        "🎛️ Interactive ROI Scenario Simulator",
        "⚡ Hourly Revenue Velocity ($/Hour)",
        "💳 Payment Disparity & Fee Decomposition",
        "🔄 Deadhead Corridor Risk Matrix"
    ])

    # -------------------------------------------------------------------------
    # TAB A: Interactive What-If ROI Simulator
    # -------------------------------------------------------------------------
    with tab_roi:
        st.subheader("Dynamic Operational ROI & Value Creation Simulator")
        st.write("Adjust operational levers below to forecast annual financial impact across the fleet (~13,500 active drivers and ~46M clean annual trips).")

        sim_col1, sim_col2 = st.columns([1, 1])
        with sim_col1:
            st.markdown("#### ⚙️ Operational Levers")
            tip_lift = st.slider("Digital Tip UI Optimization Lift (%)", min_value=0.5, max_value=5.0, value=2.5, step=0.5,
                                 help="Expected percentage point increase in tips from 20%/25%/30% POS terminal presets")
            cash_conv = st.slider("Cash-to-Digital Conversion Rate (%)", min_value=5.0, max_value=50.0, value=25.0, step=5.0,
                                  help="Share of cash passengers converted to digital payment profiles")
            deadhead_red = st.slider("Deadhead Cruising Reduction (%)", min_value=10.0, max_value=60.0, value=30.0, step=5.0,
                                     help="Reduction in empty cruising miles from dynamic outer-borough staging incentives")
            cost_per_mile = st.number_input("Driver Operating Cost per Mile ($)", min_value=0.40, max_value=1.20, value=0.65, step=0.05)

        # Run dynamic simulation
        sim_res = BusinessDecisionEngine.simulate_roi_scenarios(
            tip_lift_pct=tip_lift,
            cash_to_digital_pct=cash_conv,
            deadhead_reduction_pct=deadhead_red,
            cost_per_mile=cost_per_mile
        )

        with sim_col2:
            st.markdown("#### 📈 Projected Financial Returns")
            st.metric("Total Driver Take-Home Pay Lift", f"${sim_res['total_driver_earnings_boost']:,.2f}",
                      delta=f"+${sim_res['per_driver_annual_benefit']:,.2f} / driver annually")
            st.metric("Annual Fuel & Cruising Cost Savings", f"${sim_res['deadhead_fuel_cost_savings']:,.2f}",
                      delta=f"{sim_res['deadhead_miles_saved']:,.0f} empty miles saved")
            st.metric("Platform Processing Fee Gain (2.5%)", f"${sim_res['company_platform_ebitda_boost']:,.2f}",
                      help="Incremental gross profit for platform technology operator")

        # Waterfall Value Creation Chart
        wf_labels = ['Tip Preset UI Lift', 'Cash-to-Digital', 'Deadhead Savings', 'Total Driver Boost']
        wf_values = [
            sim_res['tip_lift_driver_gain'] / 1e6,
            sim_res['cash_conversion_tip_gain'] / 1e6,
            sim_res['deadhead_fuel_cost_savings'] / 1e6,
            sim_res['total_driver_earnings_boost'] / 1e6
        ]
        fig_wf = px.bar(
            x=wf_labels, y=wf_values,
            color=wf_labels,
            color_discrete_sequence=['#2ca02c', '#17becf', '#ff7f0e', '#1f77b4'],
            labels={'x': 'Value Creation Lever', 'y': 'Annual Benefit ($ Millions)'},
            title=f"Projected Annual Value Creation: +${sim_res['total_driver_earnings_boost']/1e6:.2f} Million Total Benefit"
        )
        fig_wf.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_wf, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB B: Hourly Revenue Velocity ($/Hour)
    # -------------------------------------------------------------------------
    with tab_velocity:
        st.subheader("Driver Revenue Velocity ($/Hour) by Borough & Diurnal Daypart")
        st.write("Tracking gross driver earnings per hour of trip duration. Congestion during Midday in Manhattan dramatically depresses hourly yield compared to outer borough highways.")

        pivot_v = velocity_df.pivot(index='borough_name', columns='daypart', values='hourly_velocity')
        daypart_order = ["Morning Rush", "Midday", "Evening Rush", "Late Night", "Overnight"]
        valid_cols = [c for c in daypart_order if c in pivot_v.columns]
        pivot_v = pivot_v[valid_cols]

        fig_heat = px.imshow(
            pivot_v,
            labels=dict(x="Diurnal Daypart", y="Origin Borough", color="Hourly Velocity ($/Hr)"),
            x=valid_cols,
            y=pivot_v.index.tolist(),
            color_continuous_scale="Viridis",
            text_auto=".1f",
            aspect="auto"
        )
        fig_heat.update_layout(height=420)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("Detailed Revenue Velocity Matrix")
        st.dataframe(
            velocity_df[['borough_name', 'daypart', 'trip_count', 'hourly_velocity', 'avg_speed_mph', 'per_mile_yield', 'avg_duration_min']].sort_values('hourly_velocity', ascending=False),
            use_container_width=True,
            hide_index=True
        )

    # -------------------------------------------------------------------------
    # TAB C: Payment Disparity & Fee Decomposition
    # -------------------------------------------------------------------------
    with tab_payment:
        st.subheader("Payment Method Disparity & Passenger Fare Decomposition")
        col_pay1, col_pay2 = st.columns(2)

        with col_pay1:
            st.markdown("#### Effective Tip % by Settlement Method")
            fig_pay = px.bar(
                pay_df.head(4),
                x='payment_label', y='effective_tip_pct',
                color='payment_label',
                labels={'payment_label': 'Payment Method', 'effective_tip_pct': 'Effective Tip % on Base Fare'},
                color_discrete_sequence=['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e']
            )
            fig_pay.update_layout(showlegend=False, height=360)
            st.plotly_chart(fig_pay, use_container_width=True)
            st.caption("Credit card rides average **26.5% tips** with 93.8% compliance, while cash rides record **0.00% system tips**.")

        with col_pay2:
            st.markdown("#### Passenger Spend Decomposition (%)")
            fee_info = kpis.get("fee_decomposition", {})
            labels = ['Driver Base Fare', 'Driver Tips', 'Non-Fare Surcharges & Taxes', 'Tolls']
            values = [
                fee_info.get('base_fare_share_pct', 68.55),
                fee_info.get('tip_share_pct', 11.03),
                fee_info.get('surcharges_share_pct', 13.23),
                fee_info.get('toll_share_pct', 1.84)
            ]
            fig_pie = px.pie(
                names=labels, values=values,
                hole=0.45,
                color_discrete_sequence=['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e']
            )
            fig_pie.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
            st.caption("Over **13.2% of customer gross spend** goes to MTA taxes, improvement surcharges, and congestion relief fees.")

    # -------------------------------------------------------------------------
    # TAB D: Deadhead Corridor Risk Matrix
    # -------------------------------------------------------------------------
    with tab_deadhead:
        st.subheader("The Deadhead Trap: Outbound vs Inbound Return Asymmetry")
        st.write("Quantifying empty cruising after dropping passengers in outer boroughs. High asymmetry ratios indicate severe uncompensated return travel.")

        clean_dh = deadhead_df[deadhead_df['dest_boro'].isin(['Queens', 'Brooklyn', 'Bronx', 'EWR', 'Staten Island'])].copy()

        fig_dh = go.Figure()
        fig_dh.add_trace(go.Bar(
            x=clean_dh['dest_boro'], y=clean_dh['outbound_volume'],
            name='Outbound from Manhattan', marker_color='#1f77b4'
        ))
        fig_dh.add_trace(go.Bar(
            x=clean_dh['dest_boro'], y=clean_dh['return_volume'],
            name='Inbound Return to Manhattan', marker_color='#ff7f0e'
        ))
        fig_dh.update_layout(barmode='group', height=400, yaxis_title="Monthly Trip Volume",
                             title="Outbound Trips vs Return Inflow from Outer Boroughs")
        st.plotly_chart(fig_dh, use_container_width=True)

        st.subheader("Corridor Deadhead Exposure Table")
        st.dataframe(
            clean_dh[['dest_boro', 'outbound_volume', 'return_volume', 'asymmetry_ratio', 'estimated_deadhead_pct', 'avg_fare', 'avg_distance']],
            use_container_width=True,
            hide_index=True
        )
        st.info("💡 **Key Finding:** Manhattan ➔ Brooklyn experiences a **59.9% deadhead rate** (>43,000 empty returns/month), while Newark Airport (EWR) experiences a **99.9% deadhead rate** due to interstate TLC licensing restrictions.")

def main():
    st.sidebar.title("🚕 DataCraft UrbanFlow")
    st.sidebar.markdown("**Datathon 2026**")
    
    navigation = st.sidebar.radio(
        "Platform Modules",
        [
            "🤖 AI Mobility Assistant",
            "🚕 Upfront Base Fare Estimator",
            "🗺️ Hotspot & OD Flow Clustering (Section 3.2)",
            "💼 Executive Decision Engine (Track 6)",
            "⏱️ ETA & Trip Duration (Teammate)",
            "📈 Demand Forecasting",
            "📊 Executive Overview & Quality"
        ]
    )
    
    zone_df = load_zone_data()
    pipeline = load_fare_pipeline()
    arrival_estimator = load_arrival_estimator()
    
    if navigation == "🤖 AI Mobility Assistant":
        render_ai_assistant_tab()
    elif navigation == "🚕 Upfront Base Fare Estimator":
        render_fare_prediction_tab(zone_df, pipeline)
    elif navigation == "🗺️ Hotspot & OD Flow Clustering (Section 3.2)":
        clusters_df, od_df, metrics = load_clustering_data()
        render_clustering_tab(zone_df, clusters_df, od_df, metrics)
    elif navigation == "💼 Executive Decision Engine (Track 6)":
        pay_df, velocity_df, deadhead_df, kpis = load_business_data()
        render_business_decision_tab(pay_df, velocity_df, deadhead_df, kpis)
    elif navigation == "⏱️ ETA & Trip Duration (Teammate)":
        render_arrival_prediction_tab(zone_df, arrival_estimator)
    elif navigation == "📈 Demand Forecasting":
        from dashboard.demand import render_demand_tab
        render_demand_tab(zone_df)
    else:
        st.header("📊 Data Quality & System Overview")
        st.markdown("### Team DataCraft Pipeline Summary")
        st.write("- **Raw Records Analyzed**: 48,599,839 rows (12 months)")
        st.write("- **Clean Dataset Size**: 45,956,110 rows (94.56% retained)")
        st.write("- **Anomalies Dropped**: 2,643,729 rows (5.44%)")
        st.write("- **Section 2.1 Upfront Base Fare Model**: LightGBM Pipeline (Test MAE: $4.60, R²: 0.68)")
        st.write("- **Section 3.2 Hotspot & Flow Clustering**: 5 Diurnal Mobility Archetypes (Silhouette: 0.2819)")
        st.write("- **Track 6 Executive Decision Engine**: +$25.97M Annual Value Simulator & Deadhead Optimization")

if __name__ == "__main__":
    main()


