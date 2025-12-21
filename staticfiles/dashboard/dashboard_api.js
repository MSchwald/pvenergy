/**
 * Universal function to load Django data into a container
 * @param {string} url - URL to Django end point
 * @param {string | array} containerId - ID of HTML container or an array of several IDs
 * @param {string} loadingHtml - Optional: Text displayed while loading data
 * @param {function} successHtml- Optional: An html for each ID provided
 * to be displayed after loading. Takes the obtained data as an argument.
 * @param {string} errorHtml - Optional: Text displayed when loading fails
 */
async function djangoRequest({
    url,
    containerId = null,
    loadingHtml = `<p class="text-loading">Loading...</i>`,
    successHtml = (data) => `<p class="text-success">Successfully loaded.</p>`,
    errorHtml = `<p class="text-error">Error while loading.</p>`,
    method = "GET",
    body = null
}) {
    const containerIds = Array.isArray(containerId) ? containerId : (containerId ? [containerId] : []);
    const containers = {};
    for (const cid of containerIds) {
        const el = document.getElementById(cid);
        if (el) {
            el.innerHTML = loadingHtml;
            containers[cid] = el;
        }
    }
    try {
        const response = await fetch(url, {
            method: method,
            headers: body ? { "Content-Type": "application/json" } : undefined,
            body: body
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        let result = successHtml(data);
        
        if (typeof result === "string") result = { [containerIds[0]]: result };
        if (Object.keys(result).length !== containerIds.length) {
            console.warn("Mismatch between containerIds and successHtml result keys.");
        }
        for (const cid of containerIds) {
            if (containers[cid] && result[cid] !== undefined) {
                containers[cid].innerHTML = result[cid];
            }
        }
        return data;
    } catch (error) {
        console.error("AJAX request failed:", error);
        for (const cid of containerIds) {
            if (containers[cid]) containers[cid].innerHTML = errorHtml;
        }
        return null;
    }
}

// Load ml models once at start
let modelsLoadedPromise = null;
function loadTrainedModels() {
    if (!modelsLoadedPromise) {
        modelsLoadedPromise = djangoRequest({
            url: "/load-models/",
            containerId: "status-container",
            loadingHtml: '<p class="text-loading">Loading trained ML models...</p>',
            successHtml: () => '<p class="text-success">ML models successfully loaded.</p>',
            errorHtml: '<p class="text-error">Failed to load ML models.</p>'
        });
    }
    return modelsLoadedPromise;
}

//When fetching OpenMeteo weather fails, do up to 3 retries
async function fetchWithRetries(url, options = {}, retries = 3) {
    for (let i = 0; i < retries; i++) {
        try {
            const res = await fetch(url, options);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return await res;
        } catch (err) {
            if (i === retries - 1) throw err;
            await new Promise(r => setTimeout(r, 500 * (i + 1)));
        }
    }
}

//Fetch OpenMeteo weather forecasts
async function fetchWeatherAndCache(containerId) {
    const info = await djangoRequest({
        url: "/load-weather/",
        containerId: containerId,
        loadingHtml: `<p class="text-loading">Checking which locations need updates...</p>`,
        successHtml: data => `<p class="text-success">Locations info loaded. ${data.locations.length} locations to fetch.</p>`,
    });
    let weatherDict = {};
    if (info && info.locations.length) {
        const url = info?.url;
        const parameters = info?.parameters;
        const locations = info?.locations || [];
        const fetchPromises = locations.map(([lat, lon]) => {
            const params = new URLSearchParams({ ...parameters, latitude: lat, longitude: lon });
            if (Array.isArray(parameters.hourly)) {
                params.set("hourly", parameters.hourly.join(","));
            }
            return fetchWithRetries(`${url}?${params.toString()}`)
                .then(r => r.json())
                .then(data => ({ [`${lat},${lon}`]: data }));
        });
        const weatherArray = await Promise.all(fetchPromises);
        weatherDict = weatherArray.reduce((acc, cur) => ({ ...acc, ...cur }), {});
    }
    await djangoRequest({
        url: "/save-weather/",
        containerId: containerId,
        loadingHtml: `<p class="text-loading">Saving fetched weather data...</p>`,
        successHtml: data => `<p class="text-success">Fetched weather data cached for ${data.count} locations.</p>`,
        method: "POST",
        body: JSON.stringify(weatherDict)
    });
}

// The formated html code to be displayed after successfully loading each request
const Success = {
    plot_weather: data => {
        const result = {};
        for (const [id, value] of Object.entries(data)) {
            result[`forecast-weather-${id}`] = `<h3>${value.weather_title}</h3>
                                                <h3> OpenMeteo weather forecast (solar radiation, air temperature, wind)</h3>
                                                <img src="${value.weather_url}" alt="Weather forecast for system ${id}">`
        }
        return result;
    },
    plot_features: data => {
        const result = {};
        for (const [id, value] of Object.entries(data)) {
            result[`forecast-features-${id}`] = `<h3> Some calculated features (solar geometry, cloudyness, cooling effects)</h3>
                                                <img src="${value.features_plot_url}" alt="Calculated features for system ${id}">`;
        }
        return result;
    },
    plot_predictions: data => {
        const result = {};
        for (const [id, value] of Object.entries(data)) {
            result[`forecast-prediction-${id}`] = `<h3> DC Power predicted by machine learning models </h3>
                                                <img src="${value.prediction_plot_url}" alt="Predicted DC Power for system ${id}">
                                                <h3> Resulting energy [kWh] (before inverters) </h3>
                                                ${value.energy}
                                                <br>
                                                <h3> Raw training features and predicted DC power </h3>
                                                ${value.df_html}`
        }
        return result;
    },
    models_training_results: data => {
        const result = {};
        for (const [id, value] of Object.entries(data)) {
            result[`model-data-${id}`] = `The model was trained on the features
                                        <h3> ${value.features} </h3>
                                        to predict the feature ${value.target}.
                                        <h3>Evaluation results</h3>
                                        ${value.evaluations}
                                        <h3>Hyperparameters test</h3>
                                        ${value.parameter}
                                        <h3>Individual system analysis</h3>
                                        ${value.system_evaluation}`
        }
        return result;
    }
};

const statusContainerId = "status-container"
async function fetchWeatherAndPipeline(ids) {
    await fetchWeatherAndCache(statusContainerId);
    await djangoRequest({
        url: "/plot-weather/",
        containerId: ids.map(id => `forecast-weather-${id}`),
        loadingHtml: '<p class="text-loading">Rendering weather plots...</p>',
        successHtml: Success.plot_weather
    });
    await djangoRequest({
        url: "/plot-features/",
        containerId: ids.map(id => `forecast-features-${id}`),
        loadingHtml: '<p class="text-loading">Calculating feature plots...</p>',
        successHtml: Success.plot_features
    });
    await loadTrainedModels();
    await djangoRequest({
        url: "/plot-predictions/",
        containerId: ids.map(id => `forecast-prediction-${id}`),
        loadingHtml: '<p class="text-loading">Predicting DC Power...</p>',
        successHtml: Success.plot_predictions
    });
}

let systemIds = []
let modelNames = null
document.addEventListener("DOMContentLoaded", async () => {
    if (!sessionStorage.getItem("metadataLoaded")) {
        sessionStorage.setItem("metadataLoaded", "true");
        loadTrainedModels()
    }
    if (window.location.pathname === "/machine_learning_models/") {
        if (!modelNames) {
            const result = await djangoRequest({
                url: "/models-names/"
            });
            modelNames = result.names
        }
        const modelContainerIds = modelNames.map((name, i) => {
            const nameContainer = document.getElementById(`model-name-${i}`);
            const dataContainer = document.getElementById(`model-data-${i}`)
            if (nameContainer) nameContainer.innerHTML = `<h2>${name}</h2>`;
            if (dataContainer) dataContainer.innerHTML = `<p class="text-loading">Loading trained ML model ${name}...</p>`;
            return i;
        });
        await loadTrainedModels();
        await djangoRequest({
            url: "/models-training-results/",
            containerId: modelContainerIds.map(i => `model-data-${i}`),
            successHtml: Success.models_training_results
        });
    }
    if (window.location.pathname === "/") {
    //if (window.location.pathname === "/" || window.location.pathname === "/all_system_forecast/") {
        const metadata = await djangoRequest({
            url: "/load-metadata/",
            containerId: "status-container",
            loadingHtml: '<p class="text-loading">Loading system metadata...</p>',
            successHtml: data => '<p class="text-success">Loading system metadata...</p>',
        });
        systemIds = Object.keys(metadata);
        for (const id of systemIds) {
            const container = document.getElementById(`system-metadata-${id}`);
            if (container) {
                container.innerHTML = metadata[id];
            }
        }
        fetchWeatherAndPipeline(systemIds);
        const now = new Date();
        const delayMs = (60 - now.getMinutes()) * 60 * 1000 - now.getSeconds() * 1000 - now.getMilliseconds();
        setTimeout(() => {
            fetchWeatherAndPipeline(systemIds);
            setInterval(() => fetchWeatherAndPipeline(systemIds), 60 * 60 * 1000);
        }, delayMs);
    }
});

/*
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
*/
