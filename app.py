import io
import base64
import traceback
import requests
from flask import Flask, render_template, jsonify, request
from datasets import load_dataset

app = Flask(__name__)
dataset = None

print("Checking/Loading SKIPP'D dataset from Hugging Face...")
try:
    dataset = load_dataset("solarbench/SKIPPD", split="train")
    print(f"Dataset loaded successfully! Total records: {len(dataset)}")
except Exception as e:
    print("CRITICAL: Failed to load dataset on startup!")
    traceback.print_exc()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/pv-data', methods=['GET'])
def get_pv_data():
    if dataset is None:
        return jsonify({"success": False, "error": "Dataset not initialized."}), 503

    try:
        try:
            start_idx = int(request.args.get('start', 0))
            limit = int(request.args.get('limit', 180))
        except ValueError:
            return jsonify({"success": False, "error": "Invalid integers for parameters."}), 400

        if start_idx < 0 or start_idx >= len(dataset):
            return jsonify({"success": False, "error": "Start index out of bounds."}), 400

        end_idx = min(start_idx + limit, len(dataset))
        data_slice = dataset[start_idx:end_idx]
        
        if 'pv' not in data_slice or 'time' not in data_slice:
            return jsonify({"success": False, "error": "Schema mismatch in backend columns."}), 500

        # Extract boundary dates for the Open-Meteo API payload
        t_start = data_slice['time'][0]
        t_end = data_slice['time'][-1]
        
        start_date = t_start.strftime('%Y-%m-%d') if hasattr(t_start, 'strftime') else str(t_start)[:10]
        end_date = t_end.strftime('%Y-%m-%d') if hasattr(t_end, 'strftime') else str(t_end)[:10]

        # Fetch matching weather data from Open-Meteo archive
        weather_lookup = {}
        try:
            weather_url = "https://archive-api.open-meteo.com/v1/archive"
            weather_params = {
                "latitude": 37.4275,       # Stanford University Latitude
                "longitude": -122.1697,    # Stanford University Longitude
                "start_date": start_date,
                "end_date": end_date,
                "hourly": "temperature_2m,cloud_cover,direct_radiation",
                "timezone": "America/Los_Angeles"
            }
            response = requests.get(weather_url, params=weather_params, timeout=5)
            if response.ok:
                hourly_data = response.json().get('hourly', {})
                # Build an efficient dictionary lookup mapping "YYYY-MM-DDTHH:00" -> Weather metrics
                for idx, time_str in enumerate(hourly_data.get('time', [])):
                    weather_lookup[time_str] = {
                        "temp": hourly_data['temperature_2m'][idx],
                        "cloud": hourly_data['cloud_cover'][idx],
                        "rad": hourly_data['direct_radiation'][idx]
                    }
        except Exception as api_err:
            print(f"Warning: Open-Meteo API retrieval down: {api_err}")

        # Assemble and join datasets
        records = []
        for i in range(len(data_slice['pv'])):
            raw_time = data_slice['time'][i]
            formatted_time = raw_time.strftime('%Y-%m-%d %H:%M') if hasattr(raw_time, 'strftime') else str(raw_time)[:16]
            
            # Create a matching string key to pull the corresponding hour's weather
            # Format: "YYYY-MM-DDTHH:00"
            hour_lookup_key = f"{formatted_time[:10]}T{formatted_time[11:13]}:00"
            weather_match = weather_lookup.get(hour_lookup_key, {"temp": 0.0, "cloud": 0, "rad": 0.0})

            records.append({
                "index": start_idx + i,
                "time": formatted_time,
                "pv_power": float(data_slice['pv'][i]),
                "temperature": weather_match["temp"],
                "cloud_cover": weather_match["cloud"],
                "radiation": weather_match["rad"]
            })
            
        return jsonify({
            "success": True,
            "total_records": len(dataset),
            "data": records
        })
        
    except Exception as e:
        traceback.print_exc() 
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/sky-image/<int:index>', methods=['GET'])
def get_sky_image(index):
    if dataset is None:
        return jsonify({"success": False, "error": "Dataset not initialized."}), 503
    try:
        if index < 0 or index >= len(dataset):
            return jsonify({"success": False, "error": "Index out of bounds."}), 400
            
        row = dataset[index]
        pil_img = row['image']
        raw_time = row['time']
        formatted_time = raw_time.strftime('%Y-%m-%d %H:%M') if hasattr(raw_time, 'strftime') else str(raw_time)[:16]

        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='JPEG')
        base64_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        return jsonify({
            "success": True,
            "image_url": f"data:image/jpeg;base64,{base64_img}",
            "time": formatted_time
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)