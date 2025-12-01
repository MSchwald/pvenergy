import sys
from pathlib import Path

from dataanalysis import Pipeline, Model, ML_MODELS, EVALUATIONS, EvaluationMethod
from data_request import Pvdaq
from feature_catalog import FeatureCatalog as F
from plotting import Plot

from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
#from .forms import ForecastForm

#print(BASE_DIR / "trained_models" / Path(file_name + ".joblib"))
#m = Model.load("xgboost")

cache = {}
SYSTEM_IDS = Pipeline.TRAINING_IDS
ml_models = []

def static_url(file: Path) -> str:
    rel = file.relative_to(settings.STATICFILES_DIRS[0])
    return f"{settings.STATIC_URL}{rel.as_posix()}"
    
def index(request):
    results = {}
    for system_id in SYSTEM_IDS:
        meta = {ftr.display_name_with_unit(): value for ftr,value in Pvdaq.meta(system_id).items() if not ftr in Pvdaq.DATA_COLUMNS}
        results[system_id] = {
            "meta": meta
        }
    return render(
        request,
        "dashboard/index.html",
        {"results": results}
    )

def plot_weather(request, system_id):
    """Schnelle Wettervorhersage"""
    df = Pipeline.weather_forecast(system_id)
    cache[system_id] = df
    weather_url = static_url(Plot.weather_forecast(df, system_id))
    return JsonResponse({"weather_plot_url": weather_url})

def plot_features(request, system_id):
    if not system_id in cache.keys():
        return JsonResponse({"error": f"Weather forecast for system {system_id} not loaded yet"}, status=400)
    df = cache[system_id]
    return JsonResponse({"features_plot_url": static_url(Plot.calculated_features(df, system_id))})

def load_models(request):
    global ml_models
    if not ml_models:
        ml_models = tuple(Model.load(ml_model.name) for ml_model in [ML_MODELS.XGBOOST, ML_MODELS.LIGHTGBM, ML_MODELS.RANDOM_FOREST])
        return JsonResponse({"status": "success", "message": "Models loaded"})
    else:
        return JsonResponse({"status": "already_loaded", "message": "Models already in memory"})

def plot_prediction(request, system_id):
    df = cache[system_id]
    if ml_models is None:
        return JsonResponse({"error": "Models not loaded yet"}, status=400)
    Y = Pipeline.predict(ml_models, df)
    return JsonResponse({"prediction_plot_url": static_url(Plot.predict(Y, system_id))})