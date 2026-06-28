import streamlit as st
import tzdata
from datasets import load_dataset
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import timedelta
from pvlib.location import Location

# --- STEP 1: UI ARCHITECTURE, PREMIUM TYPOGRAPHY & MATERIAL ICONS ---
st.set_page_config(layout="wide", page_title="PVData Studio")

custom_css = """
<style>
/* Robust Material Icons & Font Import Endpoints */
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght=400;500;600&family=Space+Grotesk:wght=500;600;700&display=swap');

/* Global Reset to Premium Typefaces */
html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
h1, h2, h3, h4, h5, h6, [data-testid="stWidgetLabel"] p {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
}

/* Material Icons Integration Styling */
.material-icons {
    font-family: 'Material Icons' !important;
    font-weight: normal;
    font-style: normal;
    font-size: 22px;
    line-height: 1;
    display: inline-block;
    text-transform: none;
    letter-spacing: normal;
    word-wrap: normal;
    white-space: nowrap;
    direction: ltr;
    vertical-align: -4px;
    margin-right: 4px;
}

/* Visual Polish for KPI Containers */
div[data-testid="stMetricContainer"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(128, 128, 128, 0.1);
    padding: 0.8rem 1rem;
    border-radius: 8px;
}

/* Dense control padding fixes */
div[data-testid="stWidgetLabel"] {
    margin-bottom: -4px !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# --- STEP 2: STABLE PIPELINE ENGINE & MEMORY CACHING ---
@st.cache_resource
def load_skippd_dataset():
    return load_dataset("solarbench/SKIPPD", split="train")

@st.cache_data
def get_dataset_timestamps():
    ds = load_dataset("solarbench/SKIPPD", split="train")
    return pd.Series(pd.to_datetime(ds['time'])).dt.tz_localize(None)

try:
    with st.spinner("Establishing telemetry connection to Hugging Face..."):
        dataset = load_skippd_dataset()
        all_times = get_dataset_timestamps()
except Exception as e:
    st.error(f"Connection to data layer failed: {e}")
    st.stop()


@st.cache_data
def fetch_historical_weather(start_date_str, end_date_str):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 37.4275, "longitude": -122.1697,
        "start_date": start_date_str, "end_date": end_date_str,
        "hourly": "temperature_2m,cloud_cover", "timezone": "America/Los_Angeles"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            hourly = response.json().get('hourly', {})
            return pd.DataFrame({
                "time_key": hourly.get('time', []),
                "temperature": hourly.get('temperature_2m', []),
                "cloud_cover": hourly.get('cloud_cover', [])
            })
    except Exception:
        pass
    return pd.DataFrame()


# --- STEP 3: CONTROL RIBBON HEADER & GROUPED SELECTION SYSTEM ---
valid_dates = sorted(list(all_times.dt.date.unique()))

if "chosen_date" not in st.session_state:
    st.session_state.chosen_date = valid_dates[0]

current_date = st.session_state.chosen_date

# Perfect Horizon Ribbon: Title, Selectors, and Nav Buttons locked on the same vertical alignment axis
header_col, selector_col, nav_col = st.columns([1.4, 1.8, 0.8], gap="small", vertical_alignment="bottom")

with header_col:
    st.markdown('<h1><span class="material-icons" style="color:#2563eb; font-size:30px;">solar_power</span>PVData Studio</h1>', unsafe_allow_html=True)
    st.caption("Stanford SKIPP'D Dataset Analysis Hub")

with selector_col:
    available_years = sorted(list(set(d.year for d in valid_dates)))
    try:
        y_default_idx = available_years.index(current_date.year)
    except ValueError:
        y_default_idx = 0
    
    col_y, col_m, col_d = st.columns(3)
    chosen_year = col_y.selectbox("Year", options=available_years, index=y_default_idx)
    
    available_months = sorted(list(set(d.month for d in valid_dates if d.year == chosen_year)))
    try:
        m_default_idx = available_months.index(current_date.month)
    except ValueError:
        m_default_idx = 0
    chosen_month = col_m.selectbox(
        "Month", 
        options=available_months, 
        index=m_default_idx,
        format_func=lambda m: datetime.date(2000, m, 1).strftime('%B')
    )
    
    available_days = sorted(list(set(d.day for d in valid_dates if d.year == chosen_year and d.month == chosen_month)))
    try:
        d_default_idx = available_days.index(current_date.day)
    except ValueError:
        d_default_idx = 0
    chosen_day = col_d.selectbox("Day", options=available_days, index=d_default_idx)
    
    constructed_date = datetime.date(chosen_year, chosen_month, chosen_day)
    if constructed_date != st.session_state.chosen_date:
        st.session_state.chosen_date = constructed_date
        st.rerun()

with nav_col:
    step_prev, step_next = st.columns(2)
    current_date_index = valid_dates.index(st.session_state.chosen_date)
    
    if step_prev.button("◀ Prev Day", width='stretch'):
        if current_date_index > 0:
            st.session_state.chosen_date = valid_dates[current_date_index - 1]
            st.rerun()
            
    if step_next.button("Next Day ▶", width='stretch'):
        if current_date_index < len(valid_dates) - 1:
            st.session_state.chosen_date = valid_dates[current_date_index + 1]
            st.rerun()


# --- STEP 4: TELEMETRY CHUNK EXTRACTOR & DYNAMIC TIMEZONE PVLIB ENGINE ---
day_mask = all_times.dt.date == st.session_state.chosen_date
start_idx, end_idx = day_mask.index[day_mask][0], day_mask.index[day_mask][-1] + 1
data_slice = dataset[int(start_idx):int(end_idx)]

df_display = pd.DataFrame({
    "time": data_slice['time'],
    "Solar Generation (kW)": data_slice['pv']
})
df_display['time'] = pd.to_datetime(df_display['time']).dt.tz_localize(None)

# DYNAMIC ALIGNMENT ENGINE: Compensates for the shifting seasonal offset observed in data logs
if st.session_state.chosen_date.month in [11, 12, 1, 2, 3]:
    target_tz = 'Etc/GMT+8'  # Fixed winter calibration
else:
    target_tz = 'Etc/GMT+6'  # 2-hour correction calibration for spring/summer logging frames

stanford_coords = Location(latitude=37.4275, longitude=-122.1697, tz=target_tz)
pvlib_timestamps = pd.DatetimeIndex(df_display['time']).tz_localize(target_tz)
clearsky_models = stanford_coords.get_clearsky(pvlib_timestamps)
df_display['Clear Sky GHI (W/m²)'] = clearsky_models['ghi'].values

# Merge historical weather values
date_str = st.session_state.chosen_date.strftime('%Y-%m-%d')
weather_df = fetch_historical_weather(date_str, date_str)

if not weather_df.empty:
    weather_df['time_key'] = pd.to_datetime(weather_df['time_key']).dt.tz_localize(None)
    df_display = pd.merge_asof(df_display.sort_values('time'), weather_df.sort_values('time_key'), left_on='time', right_on='time_key', direction='nearest')
    df_display = df_display.rename(columns={"temperature": "Temperature (°C)", "cloud_cover": "Cloud Cover (%)"})
else:
    df_display["Temperature (°C)"], df_display["Cloud Cover (%)"] = 0.0, 0


# --- STEP 5: MICRO STATE COMPLIANCE CONTROLLER ---
if "active_selected_time" not in st.session_state or st.session_state.active_selected_time.date() != st.session_state.chosen_date:
    st.session_state.active_selected_time = df_display['time'].iloc[0]


# --- STEP 6: SUMMARY METRIC HUB WITH INTEGRATED MATERIAL ICONS ---
slice_offset = (df_display['time'] - st.session_state.active_selected_time).abs().argmin()
metrics_match = df_display.iloc[slice_offset]

st.markdown("---")
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

with m_col1:
    st.markdown('<p style="font-size:13px; font-weight:600; opacity:0.8; margin-bottom:2px;"><span class="material-icons" style="color:#2563eb; font-size:16px;">schedule</span>TIME WINDOW</p>', unsafe_allow_html=True)
    st.metric(label="Time Frame", value=st.session_state.active_selected_time.strftime('%H:%M'), label_visibility="collapsed")

with m_col2:
    st.markdown('<p style="font-size:13px; font-weight:600; opacity:0.8; margin-bottom:2px;"><span class="material-icons" style="color:#10b981; font-size:16px;">bolt</span>ARRAY OUTPUT</p>', unsafe_allow_html=True)
    st.metric(label="Instantaneous Power", value=f"{metrics_match['Solar Generation (kW)']:.2f} kW", label_visibility="collapsed")

with m_col3:
    st.markdown('<p style="font-size:13px; font-weight:600; opacity:0.8; margin-bottom:2px;"><span class="material-icons" style="color:#ef4444; font-size:16px;">thermostat</span>AMBIENT TEMP</p>', unsafe_allow_html=True)
    st.metric(label="Atmospheric Temp", value=f"{metrics_match['Temperature (°C)']:.1f} °C", label_visibility="collapsed")

with m_col4:
    st.markdown('<p style="font-size:13px; font-weight:600; opacity:0.8; margin-bottom:2px;"><span class="material-icons" style="color:#64748b; font-size:16px;">cloud</span>CLOUD LAYER</p>', unsafe_allow_html=True)
    st.metric(label="Cloud Cover Density", value=f"{metrics_match['Cloud Cover (%)']:.0f}%", label_visibility="collapsed")

with m_col5:
    st.markdown('<p style="font-size:13px; font-weight:600; opacity:0.8; margin-bottom:2px;"><span class="material-icons" style="color:#f59e0b; font-size:16px;">wb_sunny</span>DAYTIME PEAK</p>', unsafe_allow_html=True)
    st.metric(label="Day Peak Generation", value=f"{df_display['Solar Generation (kW)'].max():.2f} kW", label_visibility="collapsed")
st.markdown("---")


# --- STEP 7: WORKSPACE SPLIT ---
col_analytics, col_media = st.columns([2, 1], gap="large")

with col_analytics:
    st.markdown('<h3><span class="material-icons" style="color:#10b981;">timeline</span>Synchronized Multi-Parameter Metrics</h3>', unsafe_allow_html=True)
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.14,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
    )
    
    fig.add_trace(go.Scatter(
        x=df_display['time'], y=df_display['Solar Generation (kW)'],
        name="Solar Output (kW)", mode="lines+markers",
        line=dict(color="#2563eb", width=2.5), marker=dict(size=4, opacity=0.8),
        hovertemplate="%{y:.2f} kW"
    ), row=1, col=1, secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=df_display['time'], y=df_display['Clear Sky GHI (W/m²)'],
        name="Clear Sky GHI", mode="lines+markers",
        line=dict(color="#f59e0b", width=1.5, dash="dash"), marker=dict(size=4, opacity=0),
        hovertemplate="%{y:.1f} W/m²"
    ), row=1, col=1, secondary_y=True)
    
    fig.add_trace(go.Scatter(
        x=df_display['time'], y=df_display['Cloud Cover (%)'],
        name="Cloud Cover (%)", mode="lines+markers",
        line=dict(color="#94a3b8", width=1.5), marker=dict(size=4, opacity=0),
        fill="tozeroy", fillcolor="rgba(148, 163, 184, 0.1)",
        hovertemplate="%{y:.0f}%"
    ), row=2, col=1, secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=df_display['time'], y=df_display['Temperature (°C)'],
        name="Temperature (°C)", mode="lines+markers",
        line=dict(color="#ef4444", width=2), marker=dict(size=4, opacity=0),
        hovertemplate="%{y:.1f} °C"
    ), row=2, col=1, secondary_y=True)
    
    fig.add_vline(x=st.session_state.active_selected_time, line_width=1.5, line_dash="dot", line_color="#4b5563")
    
    min_bound, max_bound = df_display['time'].min(), df_display['time'].max()
    
    # Height adjusted to 480 to sit level with the media context frame + slider row
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), height=480,
        hovermode="x unified", clickmode="event+select", dragmode="zoom",
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans")
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.12)", range=[min_bound, max_bound], minallowed=min_bound, maxallowed=max_bound)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.12)")
    
    chart_events = st.plotly_chart(fig, width='stretch', key="trends_chart", on_select="rerun")
    
    if chart_events and "selection" in chart_events and "points" in chart_events["selection"]:
        points_array = chart_events["selection"]["points"]
        if points_array:
            raw_iso_string = points_array[0].get("x")
            if raw_iso_string:
                parsed_timestamp = pd.to_datetime(raw_iso_string).tz_localize(None)
                st.session_state.active_selected_time = df_display['time'].iloc[(df_display['time'] - parsed_timestamp).abs().argmin()]
                st.rerun()

with col_media:
    st.markdown('<h3><span class="material-icons" style="color:#2563eb;">photo_camera</span>Sky Image</h3>', unsafe_allow_html=True)
    st.image(data_slice['image'][int(slice_offset)], caption=f"Stanford SkyCam Context Frame", width='stretch')
    
    scrubbed_time = st.select_slider(
        "Timeline Micro-Scrubber", options=df_display['time'],
        value=st.session_state.active_selected_time, format_func=lambda x: x.strftime('%H:%M'), label_visibility="collapsed"
    )
    if scrubbed_time != st.session_state.active_selected_time:
        st.session_state.active_selected_time = scrubbed_time
        st.rerun()