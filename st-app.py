import streamlit as st
import tzdata
from datasets import load_dataset
import requests
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

# --- STEP 1: CUSTOM STYLING & BRANDING (GOOGLE FONTS) ---
st.set_page_config(layout="wide", page_title="Stanford SKIPP'D Dataset Explorer")

custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
h1, h2, h3, h4, h5, h6, [data-testid="stWidgetLabel"] p {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("Stanford SKIPP'D Dataset Explorer")
st.caption("Short-term Solar Forecasting & Weather Context Dashboard")


# --- STEP 2: CACHE DATASET & TIMESTAMPS ---
@st.cache_resource
def load_skippd_dataset():
    """Loads the Hugging Face SKIPP'D dataset once and caches it in memory."""
    return load_dataset("solarbench/SKIPPD", split="train")

@st.cache_data
def get_dataset_timestamps():
    """Pre-extracts and normalizes timestamps for blazing-fast date-based lookups."""
    ds = load_dataset("solarbench/SKIPPD", split="train")
    # FIX: Explicitly cast to pd.Series so it reliably grants the .dt accessor
    return pd.Series(pd.to_datetime(ds['time'])).dt.tz_localize(None)

try:
    with st.spinner("Initializing Hugging Face Dataset Connection..."):
        dataset = load_skippd_dataset()
        all_times = get_dataset_timestamps()
    
    min_playable_time = all_times.iloc[0]
    max_playable_time = all_times.iloc[-1]
except Exception as e:
    st.error(f"Critical error connecting to Hugging Face: {e}")
    st.stop()


# --- STEP 3: CACHE & FETCH WEATHER ARCHIVE ---
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


# --- STEP 4: TIME-BASED SIDEBAR CONTROLS ---
if "current_time" not in st.session_state:
    st.session_state.current_time = min_playable_time

st.sidebar.header("Navigation Controls")

window_span = st.sidebar.selectbox(
    "Time Window Span (Minutes):",
    options=[60, 180, 360, 720, 1440],
    index=1 # Defaults to 180 mins (3 hours)
)

# Date Picker interface replacing raw row indexing offsets
chosen_date = st.sidebar.date_input(
    "Jump to Calendar Date:",
    value=st.session_state.current_time.to_pydatetime(),
    min_value=min_playable_time.to_pydatetime(),
    max_value=max_playable_time.to_pydatetime()
)

# If user picks a new date from the calendar widget, sync the time state
if chosen_date != st.session_state.current_time.date():
    # Maintain the current hour/minute configuration but shift the day
    combined_dt = datetime.combine(chosen_date, st.session_state.current_time.time())
    st.session_state.current_time = pd.Timestamp(combined_dt)

st.sidebar.markdown("### Chronological Steppers")
col_prev_day, col_prev_win, col_next_win, col_next_day = st.sidebar.columns(4)

if col_prev_day.button("◀◀ Day"):
    st.session_state.current_time -= pd.Timedelta(days=1)

if col_prev_win.button("◀ Prev"):
    st.session_state.current_time -= pd.Timedelta(minutes=window_span)

if col_next_win.button("Next ▶"):
    st.session_state.current_time += pd.Timedelta(minutes=window_span)

if col_next_day.button("Day ▶▶"):
    st.session_state.current_time += pd.Timedelta(days=1)

# Keep system time bounded within valid dataset dates
st.session_state.current_time = max(min_playable_time, min(max_playable_time - pd.Timedelta(minutes=window_span), st.session_state.current_time))


# --- STEP 5: VECTORIZED DATA PROCESSING ---
# Resolve exact array boundary indices instantly using binary search tree mapping
start_idx = all_times.searchsorted(st.session_state.current_time)
end_idx = start_idx + window_span
data_slice = dataset[int(start_idx):int(end_idx)]

df_display = pd.DataFrame({
    "time": data_slice['time'],
    "Solar Generation (kW)": data_slice['pv']
})
df_display['time'] = pd.to_datetime(df_display['time']).dt.tz_localize(None)

t_first = df_display['time'].iloc[0]
t_last = df_display['time'].iloc[-1]

with st.spinner("Syncing local atmospheric conditions..."):
    weather_df = fetch_historical_weather(t_first.strftime('%Y-%m-%d'), t_last.strftime('%Y-%m-%d'))

if not weather_df.empty:
    weather_df['time_key'] = pd.to_datetime(weather_df['time_key']).dt.tz_localize(None)
    df_display = df_display.sort_values('time')
    weather_df = weather_df.sort_values('time_key')
    
    df_display = pd.merge_asof(df_display, weather_df, left_on='time', right_on='time_key', direction='nearest')
    df_display = df_display.rename(columns={"temperature": "Temperature (°C)", "cloud_cover": "Cloud Cover (%)"})
else:
    df_display["Temperature (°C)"] = 0.0
    df_display["Cloud Cover (%)"] = 0

df_display['Timeline'] = df_display['time'].dt.strftime('%Y-%m-%d %H:%M')


# --- STEP 6: CAPTURE SELECTION STATE ---
if "active_selected_time" not in st.session_state or st.session_state.active_selected_time < t_first or st.session_state.active_selected_time > t_last:
    st.session_state.active_selected_time = t_first


# --- STEP 7: DASHBOARD UI LAYOUT SPLIT ---
main_layout, viewer_layout = st.columns([3, 1])

# Click selection mechanism configured specifically for temporal cross-filtering
click_selection = alt.selection_point(fields=['time'], on='click', empty=False)

with main_layout:
    st.info(f"📅 Active Window Timeline Span: **{df_display['Timeline'].iloc[0]}** to **{df_display['Timeline'].iloc[-1]}**")
    
    # Generate the shared vertical visual alignment marker
    indicator_df = pd.DataFrame({'time': [st.session_state.active_selected_time]})
    vertical_marker = alt.Chart(indicator_df).mark_rule(
        color='#f59e0b', 
        strokeWidth=2.5, 
        strokeDash=[5, 4]
    ).encode(x='time:T')
    
    # Chart 1: PV Power Solar Output
    st.subheader("Solar Generation Metrics")
    solar_base = alt.Chart(df_display).mark_line(color="#2563eb", strokeWidth=2).encode(
        x=alt.X('time:T', axis=alt.Axis(title=None, labels=False)), 
        y=alt.Y('Solar Generation (kW):Q'),
        tooltip=['Timeline', 'Solar Generation (kW)']
    )
    # Layer the metrics curve together with our vertical position guide
    solar_chart = alt.layer(solar_base, vertical_marker).add_params(click_selection).properties(height=220)
    solar_events = st.altair_chart(solar_chart, width="stretch", on_select="rerun")
    
    # Chart 2: Synchronized Weather Context Chart
    st.subheader("Synchronized Atmospheric Conditions")
    base_weather = alt.Chart(df_display).encode(x='time:T')
    cloud_line = base_weather.mark_line(color="#94a3b8", opacity=0.7).encode(y=alt.Y('Cloud Cover (%):Q'))
    temp_line = base_weather.mark_line(color="#ef4444").encode(y=alt.Y('Temperature (°C):Q'))
    
    weather_combined = alt.layer(cloud_line, temp_line).resolve_scale(y='independent')
    weather_chart = alt.layer(weather_combined, vertical_marker).add_params(click_selection).properties(height=220)
    weather_events = st.altair_chart(weather_chart, width="stretch", on_select="rerun")

# Capture dashboard interactions across both charts 
if solar_events and 'selection' in solar_events and solar_events['selection']:
    points = solar_events['selection'].get('param_1', [])
    if points and 'time' in points[0]:
        st.session_state.active_selected_time = pd.to_datetime(points[0]['time'], unit='ms')
elif weather_events and 'selection' in weather_events and weather_events['selection']:
    points = weather_events['selection'].get('param_1', [])
    if points and 'time' in points[0]:
        st.session_state.active_selected_time = pd.to_datetime(points[0]['time'], unit='ms')

with viewer_layout:
    st.subheader("Sky Camera Stream")
    
    # Chronological Scrubber completely isolated from data indices
    scrubbed_time = st.slider(
        "Scrub Window Timeline",
        min_value=t_first.to_pydatetime(),
        max_value=t_last.to_pydatetime(),
        value=st.session_state.active_selected_time.to_pydatetime(),
        format="HH:mm",
        step=timedelta(minutes=1)
    )
    st.session_state.active_selected_time = pd.Timestamp(scrubbed_time)
    
    # Find exact targeted record using the timeline index map
    target_global_index = all_times.searchsorted(st.session_state.active_selected_time)
    selected_row = dataset[int(target_global_index)]
    
    # Extract historical weather matches for metrics rendering
    metrics_match = df_display.iloc[(df_display['time'] - st.session_state.active_selected_time).abs().argmin()]
    
    st.image(
        selected_row['image'], 
        caption=f"Sky Frame Frame Capture", 
        width="stretch"
    )
    
    with st.container(border=True):
        st.markdown(f"📊 **Target Timestamp:**\n`{st.session_state.active_selected_time.strftime('%Y-%m-%d %H:%M')}`")
        st.divider()
        st.metric("Solar Output", f"{metrics_match['Solar Generation (kW)']:.2f} kW")
        st.metric("Cloud Cover", f"{metrics_match['Cloud Cover (%)']:.0f}%")
        st.metric("Temperature", f"{metrics_match['Temperature (°C)']:.1f} °C")