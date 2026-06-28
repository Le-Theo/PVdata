import streamlit as st
import tzdata
from datasets import load_dataset
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
# New high-fidelity solar physics library
from pvlib.location import Location

# --- STEP 1: UI ARCHITECTURE & MATERIAL DESIGN TYPOGRAPHY ---
st.set_page_config(layout="wide", page_title="PVData Studio")

custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght=400;500;600&family=Space+Grotesk:wght=500;600;700&family=Material+Icons&display=swap');

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

/* Material Icon Alignment Utilities */
.mi {
    font-family: 'Material Icons';
    font-weight: normal;
    font-style: normal;
    font-size: 20px;
    display: inline-block;
    line-height: 1;
    text-transform: none;
    letter-spacing: normal;
    word-wrap: normal;
    white-space: nowrap;
    direction: ltr;
    vertical-align: -4px;
    margin-right: 6px;
}

/* Visual Polish for KPI Containers */
div[data-testid="stMetricContainer"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(128, 128, 128, 0.1);
    padding: 1rem;
    border-radius: 8px;
}

/* Vertical Center Alignment for Ribbon Components */
.align-box {
    display: flex;
    flex-direction: column;
    justify-content: center;
    height: 100%;
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
    min_playable_time = all_times.iloc[0]
    max_playable_time = all_times.iloc[-1]
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


# --- STEP 3: CALENDAR MANAGEMENT & ERROR PREVENTION ---
valid_dates = sorted(list(all_times.dt.date.unique()))

if "chosen_date" not in st.session_state:
    st.session_state.chosen_date = valid_dates[0]

# Render Structured Control Ribbon
header_col1, header_col2, header_col3 = st.columns([2, 1, 1])

with header_col1:
    st.markdown('<h1><span class="mi" style="color:#2563eb; font-size:32px;">solar_power</span>PVData Studio</h1>', unsafe_allow_html=True)
    st.caption("Stanford SKIPP'D Dataset Analysis & Atmospheric Context Explorer")

with header_col2:
    st.markdown('<p style="margin-bottom:4px; font-size:14px; opacity:0.7;"><span class="mi" style="font-size:16px;">calendar_today</span>Analysis Calendar Picker</p>', unsafe_allow_html=True)
    
    # Native Calendar input element
    picked_date = st.date_input(
        "Active Analysis Date",
        value=st.session_state.chosen_date,
        min_value=min_playable_time.date(),
        max_value=max_playable_time.date(),
        label_visibility="collapsed"
    )
    
    # Validation Layer: If chosen calendar day has no data, find closest valid date and auto-snap
    if picked_date != st.session_state.chosen_date:
        if picked_date in valid_dates:
            st.session_state.chosen_date = picked_date
        else:
            nearest_date = min(valid_dates, key=lambda d: abs(d - picked_date))
            st.toast(f"No telemetry records on {picked_date}. Auto-reverting to nearest data frame: {nearest_date}", icon="📅")
            st.session_state.chosen_date = nearest_date
        st.rerun()

with header_col3:
    st.markdown('<p style="margin-bottom:4px; font-size:14px; opacity:0; pointer-events:none;">Navigation</p>', unsafe_allow_html=True)
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


# --- STEP 4: VECTORIZED TELEMETRY CHUNK EXTRACTOR & PVLIB MODELING ---
day_mask = all_times.dt.date == st.session_state.chosen_date
start_idx, end_idx = day_mask.index[day_mask][0], day_mask.index[day_mask][-1] + 1
data_slice = dataset[int(start_idx):int(end_idx)]

df_display = pd.DataFrame({
    "time": data_slice['time'],
    "Solar Generation (kW)": data_slice['pv']
})
df_display['time'] = pd.to_datetime(df_display['time']).dt.tz_localize(None)

# High-fidelity clear sky modeling via pvlib
stanford_coords = Location(latitude=37.4275, longitude=-122.1697, tz='America/Los_Angeles')
pvlib_timestamps = pd.DatetimeIndex(df_display['time']).tz_localize('America/Los_Angeles', ambiguous='NaT', nonexistent='shift_forward')
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


# --- STEP 6: MULTI-LEVEL SUMMARY METRIC HUB ---
slice_offset = (df_display['time'] - st.session_state.active_selected_time).abs().argmin()
metrics_match = df_display.iloc[slice_offset]

st.markdown("---")
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
m_col1.metric("Selected Time Frame", st.session_state.active_selected_time.strftime('%H:%M'))
m_col2.metric("Instantaneous Power", f"{metrics_match['Solar Generation (kW)']:.2f} kW")
m_col3.metric("Atmospheric Temp", f"{metrics_match['Temperature (°C)']:.1f} °C")
m_col4.metric("Cloud Cover Density", f"{metrics_match['Cloud Cover (%)']:.0f}%")
m_col5.metric("Day Peak Generation", f"{df_display['Solar Generation (kW)'].max():.2f} kW")
st.markdown("---")


# --- STEP 7: WORKSPACE SPLIT (HIGH-PERFORMANCE INTERACTIVE GRAPHICS) ---
col_analytics, col_media = st.columns([2, 1], gap="large")

with col_analytics:
    st.markdown('<h3><span class="mi" style="color:#10b981;">timeline</span>Synchronized Multi-Parameter Metrics</h3>', unsafe_allow_html=True)
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.14,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
    )
    
    # Trace 1: Solar Power Yield Curve (Row 1, Primary Axis)
    fig.add_trace(
        go.Scatter(
            x=df_display['time'], 
            y=df_display['Solar Generation (kW)'],
            name="Solar Output (kW)",
            mode="lines+markers",
            line=dict(color="#2563eb", width=2.5),
            marker=dict(size=4, opacity=0.8),
            hovertemplate="%{y:.2f} kW"
        ),
        row=1, col=1, secondary_y=False
    )
    
    # Trace 2: High-Precision Clear Sky GHI Reference Curve (Row 1, Secondary Axis)
    fig.add_trace(
        go.Scatter(
            x=df_display['time'], 
            y=df_display['Clear Sky GHI (W/m²)'],
            name="Clear Sky GHI (W/m²)",
            mode="lines",
            line=dict(color="#f59e0b", width=1.5, dash="dash"),
            hovertemplate="%{y:.1f} W/m²"
        ),
        row=1, col=1, secondary_y=True
    )
    
    # Trace 3: Cloud Cover Area Chart (Row 2, Primary Axis) - Mode changed to enable clicks
    fig.add_trace(
        go.Scatter(
            x=df_display['time'], 
            y=df_display['Cloud Cover (%)'],
            name="Cloud Cover (%)",
            mode="lines+markers",
            line=dict(color="#94a3b8", width=1.5),
            marker=dict(size=3, opacity=0), # Invisible markers to capture interaction click states
            fill="tozeroy",
            fillcolor="rgba(148, 163, 184, 0.1)",
            hovertemplate="%{y:.0f}%"
        ),
        row=2, col=1, secondary_y=False
    )
    
    # Trace 4: Ambient Temperature Profile Curve (Row 2, Secondary Axis) - Mode changed to enable clicks
    fig.add_trace(
        go.Scatter(
            x=df_display['time'], 
            y=df_display['Temperature (°C)'],
            name="Temperature (°C)",
            mode="lines+markers",
            line=dict(color="#ef4444", width=2),
            marker=dict(size=3, opacity=0), # Invisible markers to capture interaction click states
            hovertemplate="%{y:.1f} °C"
        ),
        row=2, col=1, secondary_y=True
    )
    
    # Active Frame Marker Pin
    fig.add_vline(
        x=st.session_state.active_selected_time, 
        line_width=1.5, 
        line_dash="dot", 
        line_color="#4b5563"
    )
    
    min_bound, max_bound = df_display['time'].min(), df_display['time'].max()
    
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
        hovermode="x unified", # Clean tracking tooltips without thick solid blocks
        clickmode="event+select",
        dragmode="zoom", # Zoom is now active by default
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans")
    )
    
    # Lockdown panning constraints to protect out-of-bounds rendering holes
    fig.update_xaxes(
        showgrid=True, 
        gridcolor="rgba(128,128,128,0.12)",
        showspikes=True,
        spikemode="across",
        spikethickness=1,
        spikecolor="#a1a1aa",
        spikedash="solid",
        range=[min_bound, max_bound],
        minallowed=min_bound,
        maxallowed=max_bound
    )
    
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.12)")
    fig.update_yaxes(title_text="Output (kW)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="GHI (W/m²)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Clouds (%)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Temp (°C)", row=2, col=1, secondary_y=True)
    
    chart_events = st.plotly_chart(fig, width='stretch', on_select="rerun")
    
    # Bidirectional click synchronization across both subplots
    if chart_events and "selection" in chart_events and "points" in chart_events["selection"]:
        points_array = chart_events["selection"]["points"]
        if points_array:
            raw_iso_string = points_array[0].get("x")
            if raw_iso_string:
                parsed_timestamp = pd.to_datetime(raw_iso_string).tz_localize(None)
                st.session_state.active_selected_time = df_display['time'].iloc[(df_display['time'] - parsed_timestamp).abs().argmin()]
                st.rerun()


with col_media:
    st.markdown('<h3><span class="mi" style="color:#2563eb;">photo_camera</span>Sky Imager Context</h3>', unsafe_allow_html=True)
    
    st.image(
        data_slice['image'][int(slice_offset)], 
        caption=f"Stanford SkyCam Imager Context Frame", 
        width='stretch'
    )
    
    scrubbed_time = st.select_slider(
        "Timeline Micro-Scrubber",
        options=df_display['time'],
        value=st.session_state.active_selected_time,
        format_func=lambda x: x.strftime('%H:%M'),
        label_visibility="collapsed"
    )
    
    if scrubbed_time != st.session_state.active_selected_time:
        st.session_state.active_selected_time = scrubbed_time
        st.rerun()
        
    st.caption("💡 Hint: Drag bounding boxes on the trends to investigate narrow solar windows. Click any point on either trend graph to jump to its matching video frame.")