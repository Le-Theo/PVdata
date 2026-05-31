# PVdata

**PVdata** is a real-time, interactive data analytics dashboard built with **Streamlit** to explore short-term solar forecasting data and cross-reference it with historical localized weather metrics. 

This application utilizes the **Stanford SKIPP'D (Solar Kamura Images and Photovoltaic Data) Dataset** hosted on Hugging Face and dynamically overlays synchronized atmospheric context fetched directly from the **Open-Meteo Historical Archive API**.

**Hosted on Streamlit:** https://tytec-pvdata.streamlit.app/

# Used data
- **OpenMeteo**: weather data
  - https://open-meteo.com/
- **Stanford University**: 2019 Sky Images and Photovoltaic Power Generation Dataset for Short-term Solar Forecasting 
  - Nie, Y., Li, X., Scott, A., Sun, Y., Venugopal, V., and Brandt, A. (2022). *2017-2019 Sky Images and Photovoltaic Power Generation Dataset for Short-term Solar Forecasting (Stanford Benchmark)*. Stanford Digital Repository. Available at https://purl.stanford.edu/dj417rh1007
  - https://huggingface.co/datasets/solarbench/SKIPPD
  - https://searchworks.stanford.edu/view/dj417rh1007

### Dataset metadata reference
  - Location: *Stanford University Campus, Stanford, California, USA*.
  - Timezone: `America/Los_Angeles` (US/Pacific).

# Web app based on Streamlit
![page view of app v2](src/appview_v2.png)

## 🚀 Key Features

* **Hugging Face Integration:** Seamlessly streams the `solarbench/SKIPPD` dataset (train split) directly into memory using memory-mapped Arrow data structures.
* **Open-Meteo Synchronization:** Automatically extracts geographic coordinates for Stanford University (`37.4275° N, 122.1697° W`) and pulls localized hourly cloud cover (`%`) and temperature (`°C`) for the active date range.
* **Data Alignment Engine:** Upscales hourly weather data blocks to align minute-by-minute with the high-resolution solar power array rows.
* **Interactive Timeline Synchronization (Altair):** Plotting dual-timeline interactive line charts. **Clicking on any data point along either graph** immediately triggers an update to the sky camera stream.
* **Live Sky Camera Viewer:** Synchronized previewer loading the 64x64 sky camera fisheye lens snapshot frame with detailed generation and meteorological metadata.
* **Granular Time Window controls:** Paginate by clicking predefined window sizes or day steps, change window spans (60 mins up to 24 hours), or jump directly to specific row offsets.

---

## 🛠️ Tech Stack & Dependencies

* **Frontend/App Framework:** [Streamlit](https://streamlit.io/) (v1.35.0+)
* **Data Infrastructure:** [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
* **Visualizations:** [Altair](https://altair-viz.github.io/) (Declarative statistical visualization library)
* **Data Processing:** [Pandas](https://pandas.pydata.org/)
* **HTTP Client:** [Requests](https://requests.readthedocs.io/)
* **Image Processing:** [Pillow (PIL)](https://pillow.readthedocs.io/)


## 💻 Local Installation & Setup

To run this dashboard locally, ensure you have Python 3.9+ installed, then follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Le-Theo/PVdata.git](https://github.com/Le-Theo/PVdata.git)
   cd PVdata
   ```
2. **Install the dependencies:**

    It is highly recommended to use a virtual environment (venv or conda):
    ```bash
    pip install -r requirements.txt
    ```
3. **Launch the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

The application will spin up a local server and automatically open a tab in your default browser at http://localhost:8501.


# Initial web app based on Flask (v1)
![page view of app v1](src/appview_v1.png)