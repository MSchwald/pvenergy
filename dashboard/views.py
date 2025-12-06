import sys
from pathlib import Path
import pandas as pd

from dataanalysis import Pipeline, Model, ML_MODELS, EVALUATIONS, EvaluationMethod
#from data_request import Pvdaq
from feature_catalog import FeatureCatalog as F
from feature_catalog import Feature
from feature_processing import FeatureProcessing as fp
from plotting import Plot
from typing import Any

from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
#from .forms import ForecastForm

#print(BASE_DIR / "trained_models" / Path(file_name + ".joblib"))
#m = Model.load("xgboost")

cache = {}
SYSTEM_IDS: tuple[int] = Pipeline.TRAINING_IDS
ml_models: tuple[Model] = tuple()
training_features: tuple[Feature] = tuple()

def number_format(number: float | int | Any) -> str | Any:
    if isinstance(number, pd.Timestamp):
        return number.strftime('%X')
    if not isinstance(number, (float, int)):
        return number
    s = str(number)
    if "." not in s:
        return s
    _, mantissa = s.split(".")
    if len(mantissa) <= 2:
        return s
    return f"{round(number, 2):.2f}"

def feature_format(name: str) -> str:
    if not isinstance(name, str):
        return name
    if not name in fp.ALL_FEATURE_NAMES:
        return name.replace("_", " ").title()
    return fp.FEATURE_FROM_NAME[name].display_name_with_unit

def pd_styler(data: pd.DataFrame | pd.Series) -> str:
    """Formats pandas objects with html code for displaying."""
    df = data.copy()
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    df.columns = [feature_format(col) for col in df.columns]
    df.columns.name, df.index.name = feature_format(df.index.name), None
    df_html = df.style.format(
        formatter=number_format
    ).format_index(
        number_format
    ).to_html(
        escape=False, table_attributes='class="df-table"'
    )
    return f'<div class="table-container">{df_html}</div>'

def static_url(file: Path) -> str:
    rel = file.relative_to(settings.STATICFILES_DIRS[0])
    return f"{settings.STATIC_URL}{rel.as_posix()}"
    
def individual_systems(request):
    results = {}
    for system_id in SYSTEM_IDS:
        metadata = pd_styler(Pipeline.get_system_constants().loc[system_id].to_frame().T)
        results[system_id] = {
            "metadata": metadata
        }
    return render(
        request,
        "dashboard/individual_systems.html",
        {"results": results, "active_page": "individual_systems"}
    )

def all_systems(request):
    return render(
        request,
        "dashboard/all_systems.html",
        {"metadata": pd_styler(Pipeline.get_system_constants()), "active_page": "all_systems"}
    )

def feature_database(request):
    features_df = "<style>.df-table th:last-child {text-align: left;}.df-table td:last-child {text-align: left;}</style>"
    features_df += pd_styler(
        pd.DataFrame({
            ftr.display_name: {
                "Internal name": ftr.name,
                "Source": ftr.source.value.replace("_", " ").title(),
                "Data type": ftr.data_type.__name__,
                "Is constant": ftr.is_constant,
                "Unit": ftr.unit,
                "Description": ftr.description
            } for ftr in fp.ALL_FEATURES
        }).T
    )
    return render(
        request,
        "dashboard/feature_database.html",
        {"features": features_df,
         "active_page": "feature_database"}
    )

def plot_weather(request, system_id):
    """Schnelle Wettervorhersage"""
    df = Pipeline.weather_forecast(system_id)
    cache[system_id] = df
    weather_url = static_url(Plot.weather_forecast(df, system_id))
    weather_title = f"Forecast, starting from {df.index[0]} local time (UTC{df.ftr.get_const(F.UTC_OFFSET)})"
    return JsonResponse({"weather_plot_url": weather_url, "weather_title": weather_title})

def plot_features(request, system_id):
    if not system_id in cache.keys():
        return JsonResponse({"error": f"Weather forecast for system {system_id} not loaded yet"}, status=400)
    df = cache[system_id]
    return JsonResponse({"features_plot_url": static_url(Plot.calculated_features(df, system_id))})

def load_models(request):
    global ml_models
    global training_features
    if not ml_models:
        ml_models = tuple(Model.load(ml_model.name) for ml_model in [ML_MODELS.XGBOOST, ML_MODELS.LIGHTGBM, ML_MODELS.RANDOM_FOREST])
        training_features = tuple(fp.FEATURE_FROM_NAME[name] for name in ml_models[0]._training_features)
        return JsonResponse({"status": "success", "message": "Models loaded"})
    else:
        return JsonResponse({"status": "already_loaded", "message": "Models already in memory"})

def integrate_timeseries(series: pd.Series) -> float:
    """Numeric integration of a pandas DatetimeIndex via trapezoid rule."""
    dt = series.index.to_series().diff().dt.total_seconds().fillna(0)
    return ((series + series.shift(1)) / 2 * dt / 3600000).sum()

def plot_prediction(request, system_id):
    df = cache[system_id]
    if ml_models is None:
        return JsonResponse({"error": "Models not loaded yet"}, status=400)
    Y = Pipeline.predict(ml_models, df)
    energy = pd_styler(pd.Series(data = {ml_model.name: integrate_timeseries(Y[ml_model.name]) for ml_model in ml_models}, name = "Energy [kWh]"))
    raw_data = pd.concat(
        [df.ftr.get(training_features), Y], axis = 1
    )
    return JsonResponse({
        "prediction_plot_url": static_url(Plot.predict(Y, system_id)),
        "energy": energy,
        "df_html": pd_styler(raw_data)
    })