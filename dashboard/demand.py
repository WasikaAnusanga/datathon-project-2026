"""Demand dashboard backed by aggregated pickup records."""
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data/processed/Urban_Flow_Analytics_Taxi_Clean_12Month.parquet"


@st.cache_data(show_spinner=False)
def load_demand(path, modified):
    # Aggregate in DuckDB instead of loading millions of trip records into pandas.
    with duckdb.connect() as con:
        first, last = con.execute(
            "SELECT min(pickup_timestamp), max(pickup_timestamp) FROM read_parquet(?)",
            [path],
        ).fetchone()
        if last is None:
            return pd.DataFrame(), first, last
        # Exclude the final calendar day because it may be incomplete.
        end = pd.Timestamp(last).normalize()
        start = max(pd.Timestamp(first).normalize() + pd.Timedelta(days=1), end - pd.Timedelta(days=56))
        frame = con.execute(
            """SELECT date_trunc('hour', pickup_timestamp) AS hour,
                      origin_loc_id AS loc_id, count(*) AS pickups
               FROM read_parquet(?)
               WHERE pickup_timestamp >= ? AND pickup_timestamp < ?
               GROUP BY 1, 2""", [path, start, end]
        ).df()
    return frame, start, end


def weekly_forecast(series, days=7):
    """Mean of matching weekday/hour slots, including zero-pickup hours."""
    profile = series.groupby([series.index.dayofweek, series.index.hour]).mean()
    future = pd.date_range(series.index[-1] + pd.Timedelta(hours=1), periods=days * 24, freq="h")
    return pd.Series([profile.get((t.dayofweek, t.hour), 0.0) for t in future], index=future, name="Forecast pickups")


def render_demand_tab(zone_df):
    st.header("📈 Urban Demand & Spatial Hotspots")
    st.caption("Explore recorded pickup demand and plan the next seven days using recent weekly patterns.")
    if not DATA.exists():
        st.warning("Demand data is unavailable. Add the cleaned 12-month trip dataset to data/processed to enable this dashboard.")
        return
    try:
        with st.spinner("Summarizing recent pickup demand…"):
            data, start, end = load_demand(str(DATA), DATA.stat().st_mtime_ns)
    except Exception as exc:
        st.error("Unable to load pickup demand from the cleaned dataset.")
        with st.expander("Data loading details"):
            st.code(str(exc))
        return
    if data.empty or (end - start).days < 14:
        st.warning("At least two complete weeks of pickup history are needed for this dashboard.")
        return

    zones = zone_df[["loc_id", "zone_name", "borough_name"]].drop_duplicates("loc_id")
    data = data.merge(zones, on="loc_id", how="left")
    data["borough_name"] = data["borough_name"].fillna("Unknown")
    data["zone_name"] = data["zone_name"].fillna("Zone " + data["loc_id"].astype(str))
    a, b, c = st.columns(3)
    borough = a.selectbox("Borough", ["All boroughs"] + sorted(data.borough_name.unique()), key="demand_borough")
    scoped = data if borough == "All boroughs" else data[data.borough_name == borough]
    options = scoped[["loc_id", "zone_name"]].drop_duplicates().sort_values("zone_name")
    names = dict(zip(options.loc_id, options.zone_name))
    zone = b.selectbox("Pickup zone", [None] + options.loc_id.tolist(), format_func=lambda x: "All zones" if x is None else f"{names[x]} ({x})", key="demand_zone")
    weeks = c.selectbox("History window", [2, 4, 8], index=2, format_func=lambda x: f"Last {x} weeks", key="demand_weeks")
    window_start = max(start, end - pd.Timedelta(weeks=weeks))
    scoped = scoped[scoped.hour >= window_start]
    if zone is not None:
        scoped = scoped[scoped.loc_id == zone]
    if scoped.empty:
        st.info("No pickups were recorded for these filters. Choose another zone or borough.")
        return
    hours = pd.date_range(window_start, end, freq="h", inclusive="left")
    series = scoped.groupby("hour").pickups.sum().reindex(hours, fill_value=0)
    forecast = weekly_forecast(series)
    daily = series.resample("D").sum()
    st.caption(f"Recorded history: {window_start:%d %b %Y} – {end - pd.Timedelta(days=1):%d %b %Y}. Final source day excluded to avoid partial-day totals. All times use dataset local time.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recorded pickups", f"{series.sum():,.0f}")
    m2.metric("Average daily pickups", f"{daily.mean():,.0f}")
    m3.metric("Next 7 days · baseline", f"{forecast.sum():,.0f}")
    m4.metric("Busiest hour · average", f"{series.groupby(series.index.hour).mean().idxmax():02d}:00")

    st.subheader("Pickup history & seven-day forecast")
    chart = pd.concat([daily.rename("Recorded pickups"), forecast.resample("D").sum()], axis=1)
    st.plotly_chart(px.line(chart, labels={"index": "Date", "value": "Pickups", "variable": "Series"}), use_container_width=True)
    st.info("Seasonal baseline: each future hour uses the average of the same weekday and hour in the selected history. These are estimated recorded trips, not live demand; weather, events, and unmet demand are not modeled.")
    if len(series) >= 21 * 24:
        train, actual = series.iloc[:-168], series.iloc[-168:]
        predicted = weekly_forecast(train)
        mae = (actual - predicted).abs().mean()
        st.caption(f"Held-out last-week check (trained only on earlier weeks): hourly MAE {mae:,.1f} pickups. This measures historical error, not a guarantee of future accuracy.")

    left, right = st.columns(2)
    with left:
        st.subheader("When pickups peak")
        patterns = pd.DataFrame({"pickups": series, "weekday": series.index.dayofweek, "hour": series.index.hour})
        heat = patterns.pivot_table(index="weekday", columns="hour", values="pickups", aggfunc="mean").reindex(range(7))
        heat.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="Blues", labels={"x": "Pickup hour", "y": "Weekday", "color": "Average pickups"}), use_container_width=True)
    with right:
        st.subheader("Pickup zone hotspots")
        hotspots = scoped.groupby(["loc_id", "zone_name", "borough_name"], as_index=False).pickups.sum().sort_values("pickups", ascending=False)
        top = hotspots.head(10).copy()
        top["Zone"] = top.zone_name + " (" + top.loc_id.astype(str) + ")"
        st.plotly_chart(px.bar(top.sort_values("pickups"), x="pickups", y="Zone", orientation="h", labels={"pickups": "Recorded pickups"}), use_container_width=True)
        leader = hotspots.iloc[0]
        st.caption(f"{leader.zone_name} accounts for {leader.pickups / series.sum():.1%} of pickups in this selection.")
    with st.expander("Zone demand details"):
        st.dataframe(hotspots.rename(columns={"loc_id": "Zone ID", "zone_name": "Zone", "borough_name": "Borough", "pickups": "Pickups"}), hide_index=True, use_container_width=True)
    export = forecast.rename_axis("Pickup hour").reset_index()
    st.download_button("Download hourly forecast (CSV)", export.to_csv(index=False).encode("utf-8"), "demand_forecast.csv", "text/csv")
