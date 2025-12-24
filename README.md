# Photovoltaic-Energy Forecasting

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Build Tool](https://img.shields.io/badge/Build-Hatchling-orange.svg)

## Project Description
This is an **end-to-end machine learning pipeline** that downloads photovoltaic data from [PVDAQ](https://openei.org/wiki/PVDAQ) and historical weather data from [NSRDB](https://nsrdb.nrel.gov/) and trains various machine learning models on forecasting the energy output of photovoltaic systems. The results are presented via a **Django-based dashboard**, which applies the trained models to [OpenMeteo](https://open-meteo.com/) weather forecasts and shows some plots, tables and statistical analysis of the models' performances.

## Quick Start with docker
If you have docker installed, run
```bash
docker run -it --rm -p 8000:8000 ghcr.io/mschwald/pvenergy:latest
```
and open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser to view the dashboard. This docker image includes three trained models with optimized hyperparameters for a sample set of engineered features, as well as the complete raw data from *PVDAQ* and *NSRDB* used to train them. You can also use this image to run all other commands of the underlying CLI (see the list below), but before this, first run once
```bash
docker compose up -d
```
to mount the directories containing the models and their evaluation data (otherwise, they get deleted right after training). Then, run for example
```bash
docker run -it --rm mschwald/pvenergy:latest train --features <feature1 feature2 ...>
```
with a list of feature names separated by spaces. View the entire list of implemented features via the `train --help` function and check out the *Feature Database* section of the dashboard for explanations on their definitions. When you do not need the new models anymore, use
```bash
docker compose down
```

## Alternative access of the CLI
You can also download this repository and access the CLI as follows (*recommended for LINUX/WSL2 only*):
* Ensure you have **Python 3.11** installed.
* (*Optional but recommended*) Use a virtual environment via `python -m venv venv` and then `source venv/bin/activate` (Linux/macOS bash) or `.\venv\Scripts\activate.bat` (Windows).
* Run `pip install -e .` in the root directory of this repository to install all dependencies.
* From now on, you can run `pvenergy` together with one of the commands in the list below.

## Complete list of commands of the CLI
* **`runserver`**: Opens the dashboard using Django's default development server to analyze the trained models in `data/models`; if further arguments get provided, they get directly passed to Django. Django's `runserver` is just for a convenient local presentation of a Django project before eventually deploying it on a public web server.
* **`train`**: By default, this uses the raw data in `data/merged` (if not present, it first runs **`request`**) to train the three machine learning models **XGBoost**, **LightGBM** and **Random Forest** for a preselected set of features and PVDAQ system IDs, saves the trained models in `data/models/<name>.joblib` and does some statistical analysis on the models' performances. The set of IDs and features can be customized by the additional arguments `--features <feature1 feature2 ...>` and `--ids <id1 id2 ...>`, respectively; then, this function overwrites the pre-trained models and the new models can be analyzed with the dashboard. The additional argument `--models <name1 ...>` allows for training only a subset of the three models' names *xgboost*, *lightgbm*, *random_forest*. Reducing the amount of selected IDs and models can be useful to save time when trying out to find better features for forecasting.
* **`evaluate`**: Runs an additional system-wise analysis on the three trained models in `data/models` to compare how well their performances generalizes over different photovoltaic systems. Not included in the standard analysis of the **`train`** command in order to save time. The optional arguments `--ids <id1 id2 ...>` and `--models <name1 ...>` can be provided to customize the used lists of IDs and models.
* **`pipeline`**: Runs **`train`**, **`evaluate`** and then **`runserver`** to do everything in a row. If optional arguments `--features <feature1 feature2 ...>`, `--ids <id1 id2 ...>` and `--models <name1 ...>` are provided, they get passed to the responsible commands.
* **`request`** (advanced): If the data in `data/merged` are not present, this command reproduces them by requests to *PVDAQ* and *NSRDB*. Despite using multithreading with pyarrow, this takes a long time, which is why I decided to provide the data for direct download from this repository and also included it into the corresponding docker image. Also, for requesting data from *NSRDB*, you would first need to sign up [here](https://developer.nrel.gov/signup/) to create a personal *email/api-key* pair and then set the environmental variables `NSRDB_EMAIL` and `NSRDB_API_KEY` accordingly.
* **`django`**: Passes further arguments to Django to execute arbitrary django commands. This is only relevant for developing purposes.

## Project structure and used Technical Stack
Everything is based on **Python 3.11**.
* **Data requests**: Clients in `pvcore/io` request data from *PVDAQ* (using **s3fs** and **pyarrow** for S3 buckets), *NSRDB* and *OpenMeteo* (using *requests* for http APIs).
* **Feature engineering**: The module `pvcore/feature` contains a self-written **pandas feature accessor** `.ftr` for comfortable engineering, managing and processing features while keeping the responsibilities clearly separated. Uses **pvlib**, **numpy** and linear regression for some more complicated features.
* **Machine learning**: After cleaning up the training data, the pipeline `pvcore/ml` uses **scikit_learn**, **xgboost** and **lightgbm** for machine learning.
* **Web Framework**: The dashboard in `dashboard` and `pvenergy` is written with **Django** as backend and **JavaScript** / **CSS** in frontend. **matplotlib** is used for plotting.
* **DevOps**: **Docker** to package everything and for safe development in a container.