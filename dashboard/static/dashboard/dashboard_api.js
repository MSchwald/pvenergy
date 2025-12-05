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
            <h3> ${weatherData.weather_title} </h3>
            <h3> OpenMeteo weather forecast (solar radiation, air temperature, wind)</h3>
            <img src="${weatherData.weather_plot_url}" alt="Weather forecast for system ${systemId}">
        `;
    }
    const featuresData = await djangoRequest(`/features/${systemId}/`, featuresContainerId, "Calculating features...");
    if (featuresData) {
        featuresContainer.innerHTML = `
            <h3> Some calculated features (solar geometry, cloudyness, cooling effects)</h3>
            <img src="${featuresData.features_plot_url}" alt="Calculated features for system ${systemId}">
        `;
    }
    const loadResult = await djangoRequest("/load-models/", predictionContainerId, "<p>Loading trained ML models...</p>");
    await new Promise(r => setTimeout(r, 0));   // oder requestAnimationFrame
    predictionContainer.innerHTML = "<p>Models loaded. Predicting DC Power...</p>";
    await new Promise(r => setTimeout(r, 600));

    const predictionData = await djangoRequest(`/prediction/${systemId}/`, predictionContainerId, "Predicting DC Power...");
    if (predictionData) {
        html = `
            <h3> DC Power predicted by machine learning models </h3>
            <img src="${predictionData.prediction_plot_url}" alt="Predicted DC Power for system ${systemId}">
            <h3> Resulting energy [kWh] (before inverters) </h3>
        `;
        html += "<table><tbody>";
        for (const [key, value] of Object.entries(predictionData.energy)) {
            html += `<tr><td>${key}</td><td>${value.toFixed(2)}</td></tr>`;
        }
        html += `
            </tbody></table>
            <br>
            <h3> Raw training features and predicted DC power </h3>
            ${predictionData.df_html}
        `;

        predictionContainer.innerHTML = html
        console.log(predictionContainer.energy)
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
                    btn.textContent = "Refresh system forecast";
                    break;
                default:
                    console.warn(`Unknown action: ${action}`);
            }
        });
    });
});
