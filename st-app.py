import streamlit as st
import tzdata  # Forces the environment to load missing time zone assets
from datasets import load_dataset
import requests
import pandas as pd
import altair as alt

# Set page to wide mode for a clean dashboard stretch
st.set_page_config(layout="wide", page_title="Stanford SKIPP'D Dataset Explorer")

st.title("Stanford SKIPP'D Dataset Explorer")
st.caption("Short-term Solar Forecasting & Weather Context Dashboard")

# --- STEP 1: CACHE & LOAD HUGGING FACE DATASET ---
@st.cache_resource
def load_skippd_dataset():
    """Loads the Hugging Face SKIPP'D dataset once and caches it in memory."""
    return load_dataset("solarbench/SKIPPD", split="train")

try:
    with st.spinner("Initializing Hugging Face Dataset Connection..."):
        dataset = load_skippd_dataset()
    total_records = len(dataset)
except Exception as e:
    st.error(f"Critical error connecting to Hugging Face: {e}")
    st.stop()


# --- STEP 2: CACHE & FETCH WEATHER ARCHIVE ---
@st.cache_data
def fetch_historical_weather(start_date_str, end_date_str):
    """Fetches hourly weather data from Open-Meteo for Stanford coordinates."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 37.4275,       # Stanford University
        "longitude": -122.1697,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "hourly": "temperature_2m,cloud_cover",
        "timezone": "America/Los_Angeles"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            hourly = response.json().get('hourly', {})
            df = pd.DataFrame({
                "time_key": hourly.get('time', []),
                "temperature": hourly.get('temperature_2m', []),
                "cloud_cover": hourly.get('cloud_cover', [])
            })
            return df
    except Exception as err:
        st.warning(f"Could not reach Open-Meteo Archive API: {err}")
    return pd.DataFrame()


# --- STEP 3: SIDEBAR CONTROLS & SESSION STATE ---
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

st.sidebar.header("Navigation Controls")

window_span = st.sidebar.selectbox(
    "Time Window Span (Minutes):",
    options=[60, 180, 360, 720, 1440],
    index=1 # Defaults to 180 mins (3 hours)
)

col_prev_day, col_prev_win, col_next_win, col_next_day = st.sidebar.columns(4)

if col_prev_day.button("◀◀ Day"):
    st.session_state.current_index = max(0, st.session_state.current_index - 1440)

if col_prev_win.button("◀ Prev"):
    st.session_state.current_index = max(0, st.session_state.current_index - window_span)

if col_next_win.button("Next ▶"):
    st.session_state.current_index = min(total_records - window_span, st.session_state.current_index + window_span)

if col_next_day.button("Day ▶▶"):
    st.session_state.current_index = min(total_records - window_span, st.session_state.current_index + 1440)

st.session_state.current_index = st.sidebar.number_input(
    "Exact Row Index Offset:",
    min_value=0,
    max_value=total_records - window_span,
    value=st.session_state.current_index,
    step=1
)


# --- STEP 4: DATA PROCESSOR AND ALIGNMENT LOOP ---
start_idx = int(st.session_state.current_index)
end_idx = start_idx + window_span
data_slice = dataset[start_idx:end_idx]

t_first = data_slice['time'][0]
t_last = data_slice['time'][-1]
start_date_str = t_first.strftime('%Y-%m-%d') if hasattr(t_first, 'strftime') else str(t_first)[:10]
end_date_str = t_last.strftime('%Y-%m-%d') if hasattr(t_last, 'strftime') else str(t_last)[:10]

weather_df = fetch_historical_weather(start_date_str, end_date_str)

compiled_records = []
for i in range(len(data_slice['pv'])):
    raw_time = data_slice['time'][i]
    formatted_time = raw_time.strftime('%Y-%m-%d %H:%M') if hasattr(raw_time, 'strftime') else str(raw_time)[:16]
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
        "Solar Generation (kW)": float(data_slice['pv'][i]),
        "Temperature (°C)": weather_match["temperature"],
        "Cloud Cover (%)": weather_match["cloud_cover"]
    })

df_display = pd.DataFrame(compiled_records)


# --- STEP 5: DASHBOARD UI LAYOUT SPLIT ---
main_layout, viewer_layout = st.columns([3, 1])

# We use an Altair single-point click selection tool.
click_selection = alt.selection_point(fields=['offset'], on='click', empty=False)

with main_layout:
    st.info(f"📅 Active Window Timeline Span: **{df_display['Timeline'].iloc[0]}** to **{df_display['Timeline'].iloc[-1]}**")
    
    # Chart 1: PV Power Solar Output
    st.subheader("Solar Generation Metrics")
    
    solar_chart = alt.Chart(df_display).mark_line(point=True, color="#2563eb").encode(
        x=alt.X('Timeline:N', axis=alt.Axis(labels=False, title=None)), # clean stacking
        y='Solar Generation (kW):Q',
        tooltip=['Timeline', 'Solar Generation (kW)']
    ).add_params(
        click_selection
    ).properties(height=250)
    
    # Updated: use_container_width=True -> width="stretch"
    solar_events = st.altair_chart(solar_chart, width="stretch", on_select="rerun")
    
    # Chart 2: Synchronized Weather Context Chart
    st.subheader("Synchronized Atmospheric Conditions")
    
    base_weather = alt.Chart(df_display).encode(
        x='Timeline:N',
        tooltip=['Timeline', 'Cloud Cover (%)', 'Temperature (°C)']
    )
    
    cloud_line = base_weather.mark_line(point=True, color="#94a3b8").encode(y='Cloud Cover (%):Q')
    temp_line = base_weather.mark_line(point=True, color="#ef4444").encode(y='Temperature (°C):Q')
    
    weather_chart = alt.layer(cloud_line, temp_line).resolve_scale(
        y='independent'
    ).add_params(
        click_selection
    ).properties(height=250)
    
    # Updated: use_container_width=True -> width="stretch"
    weather_events = st.altair_chart(weather_chart, width="stretch", on_select="rerun")

with viewer_layout:
    st.subheader("Sky Camera Stream")
    
    # --- STEP 6: RESOLVE ACTIVE SELECTED OFFSET ---
    selected_offset = 0
    
    # Check if a click registered on Chart 1
    if solar_events and 'selection' in solar_events and solar_events['selection']:
        selected_points = solar_events['selection'].get('param_1', [])
        if selected_points:
            selected_offset = selected_points[0].get('offset', 0)
            
    # Check if a click registered on Chart 2 instead
    elif weather_events and 'selection' in weather_events and weather_events['selection']:
        selected_points = weather_events['selection'].get('param_1', [])
        if selected_points:
            selected_offset = selected_points[0].get('offset', 0)
            
    # Manual scrubber fallback widget if no point is selected yet
    st.caption("Click data points on either timeline above, or use this slider:")
    selected_offset = st.slider(
        "Timeline Offset Slider", 
        min_value=0, 
        max_value=window_span - 1, 
        value=int(selected_offset),
        label_visibility="collapsed"
    )
    
    # Extract structural meta target variables
    target_global_index = start_idx + selected_offset
    selected_row = dataset[int(target_global_index)]
    selected_time = selected_row['time']
    time_stamp_str = selected_time.strftime('%Y-%m-%d %H:%M') if hasattr(selected_time, 'strftime') else str(selected_time)[:16]
    
    # Updated: use_container_width=True -> width="stretch"
    st.image(
        selected_row['image'], 
        caption=f"Sky Frame Thumbnail (64x64)", 
        width="stretch"
    )
    
    # Meta display card
    st.markdown(f"""
    <div style="background-color:#f1f5f9; padding: 10px; border-radius: 5px; border: 1px solid #cbd5e1; color:#1e293b;">
        <strong>Capture Time:</strong> {time_stamp_str}<br>
        <strong>Global Frame Index:</strong> #{target_global_index}<br>
        <strong>Power Output:</strong> {df_display.iloc[selected_offset]['Solar Generation (kW)']:.2f} kW<br>
        <strong>Cloud Cover:</strong> {df_display.iloc[selected_offset]['Cloud Cover (%)']:.0f}%<br>
        <strong>Temperature:</strong> {df_display.iloc[selected_offset]['Temperature (°C)']:.1f}°C
    </div>
    """, unsafe_allow_html=True)