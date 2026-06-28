import streamlit as st
import h5py
import numpy as np
import requests
import pandas as pd
import altair as alt
from PIL import Image
import io

st.set_page_config(layout="wide", page_title="PVdata: SKIPP'D Dataset Explorer")
st.title("PVdata: SKIPP'D Dataset Explorer")
st.caption("Short-term Solar Forecasting via Local HDF5 Storage")

# --- STEP 1: CACHE & LOAD LOCAL HDF5 FILE ---
HDF5_FILE_PATH = "2017_2019_images_pv_processed.hdf5"

@st.cache_resource
def open_hdf5_file(path):
    """Opens the local HDF5 binary matrix file in Read-Only mode."""
    try:
        # returns an h5py File object persistent pointer
        return h5py.File(path, "r")
    except Exception as e:
        st.error(f"Failed to read local HDF5 database structure: {e}")
        st.stop()

f_db = open_hdf5_file(HDF5_FILE_PATH)

# Dynamically calculate total framework records based on the internal matrix shape
# (Assuming typical group layout names: 'pv', 'time', 'image')
try:
    total_records = len(f_db['pv'])
except KeyError:
    st.error("Invalid HDF5 internal tree mapping. Could not resolve target database arrays.")
    st.stop()


# --- STEP 2: CACHE & FETCH WEATHER ARCHIVE ---
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
    except Exception as err:
        st.warning(f"Could not reach Open-Meteo Archive API: {err}")
    return pd.DataFrame()


# --- STEP 3: SIDEBAR CONTROLS ---
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

st.sidebar.header("Navigation Controls")
window_span = st.sidebar.selectbox("Time Window Span (Minutes):", options=[60, 180, 360, 720, 1440], index=1)

col_prev_day, col_prev_win, col_next_win, col_next_day = st.sidebar.columns(4)
if col_prev_day.button("◀◀ Day"): st.session_state.current_index = max(0, st.session_state.current_index - 1440)
if col_prev_win.button("◀ Prev"): st.session_state.current_index = max(0, st.session_state.current_index - window_span)
if col_next_win.button("Next ▶"): st.session_state.current_index = min(total_records - window_span, st.session_state.current_index + window_span)
if col_next_day.button("Day ▶▶"): st.session_state.current_index = min(total_records - window_span, st.session_state.current_index + 1440)

st.session_state.current_index = st.sidebar.number_input("Exact Row Index Offset:", min_value=0, max_value=total_records - window_span, value=st.session_state.current_index, step=1)


# --- STEP 4: DATA PROCESSOR AND ALIGNMENT LOOP ---
start_idx = int(st.session_state.current_index)
end_idx = start_idx + window_span

# Slice blocks efficiently out of the disk space via h5py arrays
pv_slice = f_db['pv'][start_idx:end_idx]
time_slice = f_db['time'][start_idx:end_idx]

# Safely catch baseline string dates
def parse_time_string(raw_val):
    if isinstance(raw_val, bytes):
        return raw_val.decode('utf-8')
    return str(raw_val)

start_date_str = parse_time_string(time_slice[0])[:10]
end_date_str = parse_time_string(time_slice[-1])[:10]

weather_df = fetch_historical_weather(start_date_str, end_date_str)

compiled_records = []
for i in range(len(pv_slice)):
    formatted_time = parse_time_string(time_slice[i])[:16]
    hour_key = f"{formatted_time[:10]}T{formatted_time[11:13]}:00"
    
    weather_match = {"temperature": 0.0, "cloud_cover": 0}
    if not weather_df.empty:
        match_row = weather_df[weather_df['time_key'] == hour_key]
        if not match_row.empty:
            weather_match["temperature"] = float(match_row.iloc[0]['temperature'])
            weather_match["cloud_cover"] = int(match_row.iloc[0]['cloud_cover'])

    compiled_records.append({
        "offset": i,
        "Timeline": formatted_time,
        "Solar Generation (kW)": float(pv_slice[i]),
        "Temperature (°C)": weather_match["temperature"],
        "Cloud Cover (%)": weather_match["cloud_cover"]
    })

df_display = pd.DataFrame(compiled_records)


# --- STEP 5: DASHBOARD UI LAYOUT SPLIT ---
main_layout, viewer_layout = st.columns([3, 1])
click_selection = alt.selection_point(fields=['offset'], on='click', empty=False)

with main_layout:
    st.info(f"📅 Active Window Timeline Span: **{df_display['Timeline'].iloc[0]}** to **{df_display['Timeline'].iloc[-1]}**")
    
    st.subheader("Solar Generation Metrics")
    solar_chart = alt.Chart(df_display).mark_line(point=True, color="#2563eb").encode(
        x=alt.X('Timeline:N', axis=alt.Axis(labels=False, title=None)),
        y='Solar Generation (kW):Q', tooltip=['Timeline', 'Solar Generation (kW)']
    ).add_params(click_selection).properties(height=250)
    
    solar_events = st.altair_chart(solar_chart, width="stretch", on_select="rerun")
    
    st.subheader("Synchronized Atmospheric Conditions")
    base_weather = alt.Chart(df_display).encode(x='Timeline:N', tooltip=['Timeline', 'Cloud Cover (%)', 'Temperature (°C)'])
    cloud_line = base_weather.mark_line(point=True, color="#94a3b8").encode(y='Cloud Cover (%):Q')
    temp_line = base_weather.mark_line(point=True, color="#ef4444").encode(y='Temperature (°C):Q')
    
    weather_chart = alt.layer(cloud_line, temp_line).resolve_scale(y='independent').add_params(click_selection).properties(height=250)
    weather_events = st.altair_chart(weather_chart, width="stretch", on_select="rerun")

with viewer_layout:
    st.subheader("Sky Camera Stream")
    
    selected_offset = 0
    if solar_events and 'selection' in solar_events and solar_events['selection']:
        selected_points = solar_events['selection'].get('param_1', [])
        if selected_points: selected_offset = selected_points[0].get('offset', 0)
    elif weather_events and 'selection' in weather_events and weather_events['selection']:
        selected_points = weather_events['selection'].get('param_1', [])
        if selected_points: selected_offset = selected_points[0].get('offset', 0)
            
    st.caption("Click data points on either timeline above, or use this slider:")
    selected_offset = st.slider("Timeline Offset Slider", min_value=0, max_value=window_span - 1, value=int(selected_offset), label_visibility="collapsed")
    
    target_global_index = start_idx + selected_offset
    
    # Process image matrix data straight out of HDF5 block indices
    raw_img_data = f_db['image'][int(target_global_index)]
    
    # If image array is raw uint8 pixel values, cast to PIL object directly
    if isinstance(raw_img_data, np.ndarray):
        img_preview = Image.fromarray(raw_img_data.astype('uint8'))
    else:
        # Fallback if stored as compressed blob byte chains
        img_preview = Image.open(io.BytesIO(raw_img_data))
        
    st.image(img_preview, caption="Sky Frame Thumbnail (64x64)", width="stretch")
    
    st.markdown(f"""
    <div style="background-color:#f1f5f9; padding: 10px; border-radius: 5px; border: 1px solid #cbd5e1; color:#1e293b;">
        <strong>Capture Time:</strong> {df_display.iloc[selected_offset]['Timeline']}<br>
        <strong>Global Frame Index:</strong> #{target_global_index}<br>
        <strong>Power Output:</strong> {df_display.iloc[selected_offset]['Solar Generation (kW)']:.2f} kW<br>
        <strong>Cloud Cover:</strong> {df_display.iloc[selected_offset]['Cloud Cover (%)']:.0f}%<br>
        <strong>Temperature:</strong> {df_display.iloc[selected_offset]['Temperature (°C)']:.1f}°C
    </div>
    """, unsafe_allow_html=True)