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
import io
from PIL import Image

# --- STEP 1: UI ARCHITECTURE, PREMIUM TYPOGRAPHY & MATERIAL ICONS ---
st.set_page_config(layout="wide", page_title="PVData Studio")

custom_css = """
<style>
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght=400;500;600&family=Space+Grotesk:wght=500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
h1, h2, h3, h4, h5, h6, [data-testid="stWidgetLabel"] p, [data-testid="stMetricLabel"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
}

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

div[data-testid="stMetricContainer"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(128, 128, 128, 0.1);
    padding: 0.8rem 1rem;
    border-radius: 8px;
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

@st.cache_resource
def get_cached_day_slice(start_idx, end_idx):
    global dataset
    return dataset[int(start_idx):int(end_idx)]

@st.cache_resource
def pre_render_day_images(_image_sequence):
    rendered_bytes_list = []
    for img in _image_sequence:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=75)
        rendered_bytes_list.append(buffer.getvalue())
    return rendered_bytes_list

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
    chosen_year = col_y.selectbox("Year", options=available_years, index=y_default_idx, help="Pick the year you want to look at from our available historical records.")
    
    available_months = sorted(list(set(d.month for d in valid_dates if d.year == chosen_year)))
    try:
        m_default_idx = available_months.index(current_date.month)
    except ValueError:
        m_default_idx = 0
    chosen_month = col_m.selectbox("Month", options=available_months, index=m_default_idx, format_func=lambda m: datetime.date(2000, m, 1).strftime('%B'), help="Pick a month. Changing this will automatically update the number of days you can select next.")
    
    available_days = sorted(list(set(d.day for d in valid_dates if d.year == chosen_year and d.month == chosen_month)))
    try:
        d_default_idx = available_days.index(current_date.day)
    except ValueError:
        d_default_idx = 0
    chosen_day = col_d.selectbox("Day", options=available_days, index=d_default_idx, help="Pick a specific day to load its matching sky pictures and minute-by-minute solar tracking details.")
    
    constructed_date = datetime.date(chosen_year, chosen_month, chosen_day)
    if constructed_date != st.session_state.chosen_date:
        st.session_state.chosen_date = constructed_date
        st.rerun()

with nav_col:
    step_prev, step_next = st.columns(2)
    current_date_index = valid_dates.index(st.session_state.chosen_date)
    
    if step_prev.button("◀ Prev Day", width='stretch', help="Go backward by exactly one calendar day."):
        if current_date_index > 0:
            st.session_state.chosen_date = valid_dates[current_date_index - 1]
            st.rerun()
            
    if step_next.button("Next Day ▶", width='stretch', help="Go forward by exactly one calendar day."):
        if current_date_index < len(valid_dates) - 1:
            st.session_state.chosen_date = valid_dates[current_date_index + 1]
            st.rerun()


# --- STEP 4: PRE-FETCH & INTERPOLATE DATA SPANS (RUNS ONCE PER DATE CHANGE) ---
day_mask = all_times.dt.date == st.session_state.chosen_date
start_idx, end_idx = day_mask.index[day_mask][0], day_mask.index[day_mask][-1] + 1

data_slice = get_cached_day_slice(start_idx, end_idx)

with st.spinner("Pre-rendering daily sky images into high-speed memory cache..."):
    pre_rendered_images = pre_render_day_images(data_slice['image'])

df_display = pd.DataFrame({
    "time": data_slice['time'],
    "Solar Generation (kW)": data_slice['pv']
})
df_display['time'] = pd.to_datetime(df_display['time']).dt.tz_localize(None)

if st.session_state.chosen_date.month in [12, 1, 2, 3]:
    target_tz = 'Etc/GMT+8'
else:
    target_tz = 'Etc/GMT+6'

stanford_coords = Location(latitude=37.4275, longitude=-122.1697, tz=target_tz)
pvlib_timestamps = pd.DatetimeIndex(df_display['time']).tz_localize(target_tz)
clearsky_models = stanford_coords.get_clearsky(pvlib_timestamps)
df_display['Clear Sky GHI (W/m²)'] = clearsky_models['ghi'].values

date_str = st.session_state.chosen_date.strftime('%Y-%m-%d')
weather_df = fetch_historical_weather(date_str, date_str)

if not weather_df.empty:
    weather_df['time_key'] = pd.to_datetime(weather_df['time_key']).dt.tz_localize(None)
    weather_df = weather_df.rename(columns={"temperature": "Temperature (°C)", "cloud_cover": "Cloud Cover (%)", "time_key": "time"})
    target_times = df_display['time'].copy()
    df_combined = pd.merge(df_display, weather_df, on='time', how='outer').sort_values('time')
    df_combined["Temperature (°C)"] = df_combined["Temperature (°C)"].interpolate(method='linear', limit_direction='both')
    df_combined["Cloud Cover (%)"] = df_combined["Cloud Cover (%)"].interpolate(method='linear', limit_direction='both')
    df_display = df_combined[df_combined['time'].isin(target_times)].copy()
else:
    df_display["Temperature (°C)"], df_display["Cloud Cover (%)"] = 0.0, 0

if "active_selected_time" not in st.session_state or st.session_state.active_selected_time.date() != st.session_state.chosen_date:
    st.session_state.active_selected_time = df_display['time'].iloc[0]


# --- STEP 5: MEDIA COLUMN ---
@st.fragment
def render_interactive_workspace(df_data, img_cache):
    slice_offset = (df_data['time'] - st.session_state.active_selected_time).abs().argmin()
    metrics_match = df_data.iloc[slice_offset]
    
    st.markdown("---")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    
    with m_col1:
        st.metric(label="SELECTED TIME", value=st.session_state.active_selected_time.strftime('%H:%M'),
                  help="The specific minute currently chosen. You can click anywhere on the graphs or slide the timeline scrubber below to change this.")
    with m_col2:
        st.metric(label="SOLAR GENERATION", value=f"{metrics_match['Solar Generation (kW)']:.2f} kW",
                  help="The actual electricity power being produced by the rooftop solar panels at this exact minute.")
    with m_col3:
        st.metric(label="OUTDOOR TEMPERATURE", value=f"{metrics_match['Temperature (°C)']:.1f} °C",
                  help="The outside air temperature. Hourly weather records are automatically smoothed out to guess the exact temperature for this minute.")
    with m_col4:
        st.metric(label="CLOUD COVERAGE", value=f"{metrics_match['Cloud Cover (%)']:.0f}%",
                  help="How much of the sky is covered by clouds. 0% means a perfectly clear sky, while 100% means it is completely cloudy.")
    with m_col5:
        st.metric(label="DAILY HIGHEST PEAK", value=f"{df_data['Solar Generation (kW)'].max():.2f} kW",
                  help="The absolute highest power generation spike recorded on this day. Use this to see how sunny the best part of the day was.")
    st.markdown("---")
    
    col_analytics, col_media = st.columns([2, 1], gap="large")
    
    with col_analytics:
        st.markdown(
            '<h3><span class="material-icons" style="color:#10b981;">timeline</span>Solar Metrics & OpenMeteo Weather Data</h3>', 
            unsafe_allow_html=True,
            help="Interactive graphs showing the day's trends. Click anywhere on any line to update the snapshot photo on the right."
        )
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
                            specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
        
        fig.add_trace(go.Scatter(x=df_data['time'], y=df_data['Solar Generation (kW)'], name="Solar Output (kW)", 
                                 mode="lines+markers", line=dict(color="#2563eb", width=2.5), marker=dict(size=1, opacity=0.8)), row=1, col=1, secondary_y=False)
        
        fig.add_trace(go.Scatter(x=df_data['time'], y=df_data['Clear Sky GHI (W/m²)'], name="Clear Sky GHI", 
                                 mode="lines+markers", line=dict(color="#f59e0b", width=1.5, dash="dash"), marker=dict(size=1, opacity=0)), row=1, col=1, secondary_y=True)
        
        fig.add_trace(go.Scatter(x=df_data['time'], y=df_data['Cloud Cover (%)'], name="Cloud Cover (%)", 
                                 mode="lines+markers", line=dict(color="#94a3b8", width=1.5), marker=dict(size=1, opacity=0.01), fill="tozeroy", fillcolor="rgba(148, 163, 184, 0.08)"), row=2, col=1, secondary_y=False)
        
        fig.add_trace(go.Scatter(x=df_data['time'], y=df_data['Temperature (°C)'], name="Temperature (°C)", 
                                 mode="lines+markers", line=dict(color="#ef4444", width=2), marker=dict(size=1, opacity=0.01)), row=2, col=1, secondary_y=True)
        
        fig.add_vline(x=st.session_state.active_selected_time, line_width=1.5, line_dash="dot", line_color="#4b5563")
        
        min_bound, max_bound = df_data['time'].min(), df_data['time'].max()
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=515, hovermode="x unified", clickmode="event+select", dragmode="zoom",
                          showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans"))
        
        fig.update_xaxes(title_text="Timeline", showgrid=True, gridcolor="rgba(128,128,128,0.12)", range=[min_bound, max_bound], minallowed=min_bound, maxallowed=max_bound, row=2, col=1)
        fig.update_yaxes(title_text="Solar Generation (kW)", showgrid=True, gridcolor="rgba(128,128,128,0.12)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Clear Sky GHI (W/m²)", showgrid=False, row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Cloud Coverage (%)", showgrid=True, gridcolor="rgba(128,128,128,0.12)", range=[0, 100], row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Temperature (°C)", showgrid=False, row=2, col=1, secondary_y=True)
        
        chart_events = st.plotly_chart(fig, width='stretch', key="trends_chart", on_select="rerun")
        
        if chart_events and "selection" in chart_events and "points" in chart_events["selection"]:
            points_array = chart_events["selection"]["points"]
            if points_array:
                raw_iso_string = points_array[0].get("x")
                if raw_iso_string:
                    parsed_timestamp = pd.to_datetime(raw_iso_string).tz_localize(None)
                    st.session_state.active_selected_time = df_data['time'].iloc[(df_data['time'] - parsed_timestamp).abs().argmin()]
                    st.rerun()
                    
    with col_media:
        st.markdown(
            '<h3><span class="material-icons" style="color:#2563eb;">photo_camera</span>Sky Image</h3>', 
            unsafe_allow_html=True,
            help="A photo taken by an upward-pointing fisheye lens camera mounted directly next to the solar panels. This shows you real-time clouds moving over the site."
        )
        
        st.image(img_cache[int(slice_offset)], caption=f"Stanford SkyCam Frame", width='stretch')
        
        scrubbed_time = st.select_slider(
            "Timeline Micro-Scrubber", options=df_data['time'], value=st.session_state.active_selected_time, format_func=lambda x: x.strftime('%H:%M'), label_visibility="collapsed",
            help="Drag this slider to scrub forward and backward through the pictures frame-by-frame."
        )
        if scrubbed_time != st.session_state.active_selected_time:
            st.session_state.active_selected_time = scrubbed_time
            st.rerun()
            
        img_prev_col, img_next_col = st.columns(2)
        with img_prev_col:
            if st.button("◀ Previous Frame", width='stretch', help="Step backward by exactly one image frame (1 minute)."):
                if slice_offset > 0:
                    st.session_state.active_selected_time = df_data['time'].iloc[int(slice_offset) - 1]
                    st.rerun()
        with img_next_col:
            if st.button("Next Frame ▶", width='stretch', help="Step forward by exactly one image frame (1 minute)."):
                if slice_offset < len(df_data) - 1:
                    st.session_state.active_selected_time = df_data['time'].iloc[int(slice_offset) + 1]
                    st.rerun()

    st.markdown("#### How to Read This Chart")
    guide_col1, guide_col2 = st.columns(2)
    with guide_col1:
        st.markdown("""
        **Top Chart: Solar Activity**
        * **Timeline** (x-axis): Shows the progression of minutes throughout the selected day, moving from morning on the left to evening on the right.
        * **Photovoltaic Generation** (left y-axis): Measures the actual electricity being generated by the panels in kilowatts (kW). 
        * **Clear Sky GHI** (right y-axis): Measures raw sunshine power striking the ground on a theoretical day with no clouds.
        """)
    with guide_col2:
        st.markdown("""
        **Bottom Chart: Weather Trends**
        * **Cloud Coverage** (left y-axis): Measures the cloud blanket blocking the sky from 0% to 100%.
        * **Temperature** (right y-axis): Tracks changes in ambient outside air temperature in degrees Celsius (°C).
        * **Individual Data Points:** Every point on these lines represents a 1-minute logging event. Clicking any point draws a dark vertical indicator line across both charts and instantly refreshes the sky image to match that moment.
        """)

# Run the pipeline
render_interactive_workspace(df_display, pre_rendered_images)