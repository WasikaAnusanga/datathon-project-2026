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

def main():
    st.sidebar.title("🚕 DataCraft UrbanFlow")
    st.sidebar.markdown("**Datathon 2026**")
    
    navigation = st.sidebar.radio(
        "Platform Modules",
        [
            "🤖 AI Mobility Assistant",
            "🚕 Upfront Base Fare Estimator",
            "⏱️ ETA & Trip Duration (Teammate)",
            "📈 Demand Forecasting (Teammate)",
            "📊 Executive Overview & Quality"
        ]
    )
    
    zone_df = load_zone_data()
    pipeline = load_fare_pipeline()
    
    if navigation == "🤖 AI Mobility Assistant":
        render_ai_assistant_tab()
    elif navigation == "🚕 Upfront Base Fare Estimator":
        render_fare_prediction_tab(zone_df, pipeline)
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

if __name__ == "__main__":
    main()
