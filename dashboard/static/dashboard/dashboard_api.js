/**
 * Universal function to load Django data into a container
 * @param {string} url - URL to Django end point
 * @param {string} containerId - ID of HTML container
 * @param {string} loadingText - Optional: Text displayed while loading data
 */
async function djangoRequest(url, containerId, loadingText = "Loading...") {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `<p>${loadingText}</p>`;
    }
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error("AJAX request failed:", error);
        if (container) {
            container.innerHTML = "<p>Error while loading.</p>";
        }
        return null;
    }
}

// Load weather first, then calculated features, then dcp predictions
async function loadForecast(systemId) {
    const weatherContainerId = `forecast-weather-${systemId}`;
    const featuresContainerId = `forecast-features-${systemId}`;
    const predictionContainerId = `forecast-prediction-${systemId}`;
    const weatherContainer = document.getElementById(weatherContainerId);
    const featuresContainer = document.getElementById(featuresContainerId);
    const predictionContainer = document.getElementById(predictionContainerId);

    const weatherData = await djangoRequest(`/weather/${systemId}/`, weatherContainerId, "Requesting weather forecast...");
    if (weatherData) {
        weatherContainer.innerHTML = `
            <img src="${weatherData.weather_plot_url}" alt="Weather forecast for system ${systemId}">
        `;
    }
    const featuresData = await djangoRequest(`/features/${systemId}/`, featuresContainerId, "Calculating features...");
    if (featuresData) {
        featuresContainer.innerHTML = `
            <img src="${featuresData.features_plot_url}" alt="Calculated features for system ${systemId}">
        `;
    }
    const loadResult = await djangoRequest("/load-models/", predictionContainerId, "<p>Loading trained ML models...</p>");
    await new Promise(r => setTimeout(r, 0));   // oder requestAnimationFrame
    predictionContainer.innerHTML = "<p>Models loaded. Predicting DC Power...</p>";
    await new Promise(r => setTimeout(r, 600));

    const predictionData = await djangoRequest(`/prediction/${systemId}/`, predictionContainerId, "Predicting DC Power...");
    if (predictionData) {
        predictionContainer.innerHTML = `
            <img src="${predictionData.prediction_plot_url}" alt="Predicted DC Power for system ${systemId}">
        `;
    }
}

async function loadPrediction(systemId) {
    const containerId = `predict-result-${systemId}`;
    const data = await djangoRequest(`/predict/${systemId}/`, containerId, "Predicting DC power...");

    if (data) {
        document.getElementById(containerId).innerHTML = `
            <pre>${JSON.stringify(data, null, 2)}</pre>
        `;
    }
}

// Initialize buttons
document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("button[data-action]").forEach(btn => {
        btn.addEventListener("click", () => {
            const systemId = btn.getAttribute("data-id");
            const action = btn.getAttribute("data-action");

            switch(action) {
                case "forecast":
                    loadForecast(systemId);
                    break;
                // Weitere Aktionen hier hinzufügen
                default:
                    console.warn(`Unknown action: ${action}`);
            }
        });
    });
});
