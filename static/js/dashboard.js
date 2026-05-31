document.addEventListener('DOMContentLoaded', () => {
    console.log("[Dashboard] Initializing synchronized multi-chart layout...");

    let currentIndex = 0;
    let limit = 180; 
    let totalRecordsMax = 500000; 
    let currentDataBatch = []; 
    let pvChart = null;
    let weatherChart = null; // Second Chart Instance Anchor

    // Core Controls
    const rangeLabel = document.getElementById('range-label');
    const pvCanvas = document.getElementById('pvChart');
    const weatherCanvas = document.getElementById('weatherChart');
    
    // Optional Navigation Elements
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const prevDayBtn = document.getElementById('prev-day-btn');
    const nextDayBtn = document.getElementById('next-day-btn');
    const spanSelect = document.getElementById('span-select');
    
    // Image Viewer Elements
    const skyImage = document.getElementById('sky-image');
    const placeholder = document.getElementById('image-placeholder');
    const imageMeta = document.getElementById('image-meta');

    if (!rangeLabel || !pvCanvas || !weatherCanvas) {
        console.error("[Dashboard] Core chart rendering elements missing.");
        return;
    }

    async function fetchChartData(start) {
        rangeLabel.textContent = "Loading data streams...";
        try {
            const response = await fetch(`/api/pv-data?start=${start}&limit=${limit}`);
            const result = await response.json();
            
            if (response.ok && result.success) {
                currentDataBatch = result.data;
                totalRecordsMax = result.total_records;
                currentIndex = start;
                
                // Render both charts using the same batch
                renderSolarChart(result.data);
                renderWeatherChart(result.data);
                
                if (result.data.length > 0) {
                    const startTime = result.data[0].time;
                    const endTime = result.data[result.data.length - 1].time;
                    const endTimeString = endTime.includes(' ') ? endTime.split(' ')[1] : endTime;
                    rangeLabel.textContent = `${startTime} to ${endTimeString}`;
                } else {
                    rangeLabel.textContent = "Empty Range Selection";
                }
            } else {
                rangeLabel.textContent = "Error";
                alert(`Backend Error: ${result.error}`);
            }
        } catch (error) {
            rangeLabel.textContent = "Network Error";
            console.error("[Dashboard] Connection exception:", error);
        }
    }

    function renderSolarChart(data) {
        const times = data.map(item => item.time);
        const values = data.map(item => item.pv_power);

        if (pvChart) {
            pvChart.data.labels = times;
            pvChart.data.datasets[0].data = values;
            pvChart.update();
            return;
        }

        pvChart = new Chart(pvCanvas.getContext('2d'), {
            type: 'line',
            data: {
                labels: times,
                datasets: [{
                    label: 'PV Power Output (kW)',
                    data: values,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.05)',
                    borderWidth: 2,
                    tension: 0.15,
                    pointRadius: 1,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                onClick: (e, activeElements) => {
                    handleChartClick(activeElements);
                },
                scales: {
                    x: { display: false }, // Hide X-axis line to keep stacked view clean
                    y: { title: { display: true, text: 'Solar Output (kW)' }, beginAtZero: true }
                }
            }
        });
    }

    function renderWeatherChart(data) {
        const times = data.map(item => item.time);
        const clouds = data.map(item => item.cloud_cover);
        const temps = data.map(item => item.temperature);

        if (weatherChart) {
            weatherChart.data.labels = times;
            weatherChart.data.datasets[0].data = clouds;
            weatherChart.data.datasets[1].data = temps;
            weatherChart.update();
            return;
        }

        weatherChart = new Chart(weatherCanvas.getContext('2d'), {
            type: 'line',
            data: {
                labels: times,
                datasets: [
                    {
                        label: 'Cloud Cover (%)',
                        data: clouds,
                        borderColor: '#94a3b8',
                        backgroundColor: 'rgba(148, 163, 184, 0.1)',
                        borderWidth: 2,
                        yAxisID: 'yCloud',
                        tension: 0.1
                    },
                    {
                        label: 'Temperature (°C)',
                        data: temps,
                        borderColor: '#ef4444',
                        backgroundColor: 'transparent',
                        borderWidth: 1.5,
                        yAxisID: 'yTemp',
                        tension: 0.1,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                onClick: (e, activeElements) => {
                    handleChartClick(activeElements);
                },
                scales: {
                    x: { title: { display: true, text: 'Timeline (US/Pacific)' } },
                    yCloud: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: { display: true, text: 'Cloud Cover (%)' },
                        min: 0,
                        max: 100
                    },
                    yTemp: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: { display: true, text: 'Temperature (°C)' },
                        grid: { drawOnChartArea: false } // Avoid messy overlapping grid lines
                    }
                }
            }
        });
    }

    function handleChartClick(activeElements) {
        if (activeElements.length > 0) {
            const batchIndex = activeElements[0].index;
            const globalTarget = currentDataBatch[batchIndex];
            if (globalTarget && skyImage) {
                loadSkyImage(globalTarget.index);
            }
        }
    }

    async function loadSkyImage(globalIndex) {
        if (!skyImage || !placeholder) return;
        placeholder.textContent = "Streaming sky image...";
        skyImage.style.display = 'none';
        
        try {
            const response = await fetch(`/api/sky-image/${globalIndex}`);
            const result = await response.json();
            
            if (response.ok && result.success) {
                placeholder.style.display = 'none';
                skyImage.src = result.image_url;
                skyImage.style.display = 'block';
                if (imageMeta) {
                    imageMeta.innerHTML = `<strong>Capture:</strong> ${result.time} <br><small>(Frame #${globalIndex})</small>`;
                }
            } else {
                placeholder.textContent = "Error loading frame.";
            }
        } catch (error) {
            placeholder.textContent = "Network error.";
        }
    }

    // Attach Event Observers safely if elements are verified
    if (spanSelect) {
        spanSelect.addEventListener('change', (e) => {
            limit = parseInt(e.target.value, 10);
            fetchChartData(currentIndex);
        });
    }
    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            fetchChartData(Math.max(0, currentIndex - limit));
        });
    }
    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            fetchChartData(Math.min(totalRecordsMax - limit, currentIndex + limit));
        });
    }
    if (prevDayBtn) {
        prevDayBtn.addEventListener('click', () => {
            fetchChartData(Math.max(0, currentIndex - 1440));
        });
    }
    if (nextDayBtn) {
        nextDayBtn.addEventListener('click', () => {
            fetchChartData(Math.min(totalRecordsMax - limit, currentIndex + 1440));
        });
    }

    // Run baseline load
    fetchChartData(0);
});