from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pvcore.feature import Feature

import pandas as pd
import json

from pvcore.ml import Pipeline, Model, ML_MODELS
from pvcore.io import Pvdaq, OpenMeteo
import pvcore.utils.file_utilities as fu
from pvcore.feature import Catalog as F, ALL_FEATURES, FEATURE_FROM_NAME
from pvcore.plotting import Plot
from .formatting import feature_format, pd_styler, file_to_url

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

weather_cache = {}
cache = {}
location_from_id = {}
locations = tuple()
system_constants = {}

# Lazy loading of data from the main program
def get_system_ids() -> tuple[int]:
    return Pipeline.TRAINING_IDS

# Get loaded while already rendering in order to not delay showing the starting page for several seconds
TRAINED_MODELS = [ML_MODELS.XGBOOST, ML_MODELS.LIGHTGBM, ML_MODELS.RANDOM_FOREST]
ml_models: tuple[Model] = tuple()
training_features: tuple[Feature] = (
            F.POWER_RATIO, F.PVLIB_POA_IRRADIANCE,
            F.DAY_OF_YEAR, F.TIME_SINCE_SUNLIGHT,
            F.CLEAR_SKY_RATIO, F.COS_AOI, F.WIND_NORMAL_COMPONENT,
            F.POA_COS_AOI, F.POA_WIND_SPEED, F.DHI_PER_GHI,
            F.DCP_PER_AREA, F.GAMMA_TEMP_DIFFERENCE, F.RELATIVE_AZIMUTH
)

class TemplateViews:
    def individual_systems(request):
        return {"ids": get_system_ids()}

    def all_systems(request):
        #df = pd.read_csv(RESULTS_DIR / "results.csv", index_col = 0).map(lambda x: feature_format(x, display_unit = False))
        #df.index.name = F.SYSTEM_ID.name
        return {
            "metadata": pd_styler(Pipeline.get_system_constants()),
            #"evaluations": pd_styler(df)
        }

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
                } for ftr in ALL_FEATURES
            }).T
        )
        return {"features": features_df}

    def models_info(request):
        return {"indices": list(range(len(TRAINED_MODELS)))}

class ApiEndpoints:
    def models_names(request):
        return JsonResponse({"names": [feature_format(model.name, display_unit = False) for model in TRAINED_MODELS]})

    def models_training_results(request):
        for model in ml_models:
            model._evaluation_results.name = "Score"
        system_evaluation = {}
        for model in ml_models:
            df = Pipeline.system_evaluations(model, evaluate = False)
            if df.empty:
                system_evaluation[model.name] = "Model has not yet been evaluated on each individual system. Use 'pvenergy evaluate' to get such an analysis."
            else:
                system_evaluation[model.name] = pd_styler(df)
        return JsonResponse({
            str(i): {
                    "features": ", ".join([feature_format(name, display_unit = False) for name in model._training_features]),
                    "target": feature_format(model._target_feature, display_unit = False),
                    "evaluations": pd_styler(model._evaluation_results),
                    "parameter": pd_styler(pd.Series(model.get_hyperparameters(), name="Value")),
                    "system_evaluation": system_evaluation[model.name]
            } for i, model in enumerate(ml_models)
        })

    def load_metadata(request):
        return JsonResponse({str(id): pd_styler(Pipeline.get_system_constants().loc[id].to_frame().T) for id in get_system_ids()})

    def load_weather(request):
        global locations
        if not locations:
           locations = tuple({(Pvdaq.meta(id)[F.LATITUDE], Pvdaq.meta(id)[F.LONGITUDE]) for id in get_system_ids()})
        fetch_locations = [
            (lat, lon) for (lat, lon) in locations if not fu.file_up_to_date(
                OpenMeteo.cache_name(lat, lon)
            )
        ]
        for loc in [loc for loc in locations if not loc in fetch_locations]:
            if not loc in weather_cache.keys():
                weather_cache[loc] = OpenMeteo.load_cache(OpenMeteo.cache_name(loc[0], loc[1]))
        return JsonResponse({
            "locations": fetch_locations,
            "url": OpenMeteo.url,
            "parameters": OpenMeteo.PARAMETERS
        })

    def load_models(request):
        global ml_models
        if not ml_models:
            ml_models = tuple(Model.load(ml_model.name) for ml_model in TRAINED_MODELS)
            return JsonResponse({"status": "success", "message": "Models loaded"})
        else:
            return JsonResponse({"status": "already_loaded", "message": "Models already in memory"})

    @csrf_exempt
    def save_weather(request):
        data_dict = json.loads(request.body)  # {"lat,lon": {...}, ...}
        count = 0
        for key, weather_json in data_dict.items():
            lat_str, lon_str = key.split(",")
            lat, lon = float(lat_str), float(lon_str)
            df = OpenMeteo.format_response(weather_json)
            df.to_csv(OpenMeteo.cache_name(lat, lon), index=True)
            weather_cache[(lat, lon)] = df
            count += 1
        return JsonResponse({"count": count})

    def plot_weather(request):
        """Plot weather forecasts"""
        global system_constants
        if not system_constants:
            system_constants = {id: Pipeline.system_constants(id) for id in get_system_ids()}
        global location_from_id
        if not location_from_id:
            location_from_id = {id : (Pvdaq.meta(id)[F.LATITUDE], Pvdaq.meta(id)[F.LONGITUDE]) for id in get_system_ids()}
        plots = {}
        for id in get_system_ids():
            df = weather_cache[location_from_id[id]].copy()
            df.ftr.set_const(system_constants[id])
            df = Pipeline.utc_to_local_time(df)
            df.ftr.set_const(system_constants[id])
            cache[id] = df
            plots[str(id)] = {
                "weather_url": file_to_url(Plot.weather_forecast(df, id)),
                "weather_title": f"Forecast, starting from {df.index[0]} local time (UTC{df.ftr.get_const(F.UTC_OFFSET)})"
            }
        return JsonResponse(plots)

    def plot_features(request):
        plots = {}
        for id in get_system_ids():
            df = cache[id]
            plots[str(id)] = {
                "features_plot_url": file_to_url(Plot.calculated_features(df, id))
            }
        return JsonResponse(plots)

    def plot_predictions(request):
        results = {}
        for id in get_system_ids():
            df = cache[id]
            Y = Pipeline.predict(ml_models, df)
            energy = pd_styler(pd.Series(data = {ml_model.name: Pipeline.integrate_timeseries(Y[ml_model.name]) for ml_model in ml_models}, name = "Energy [kWh]"))
            raw_data = pd.concat(
                [df.ftr.get(training_features), Y], axis = 1
            )
            results[id] = {
                "prediction_plot_url": file_to_url(Plot.predict(Y, id)),
                "energy": energy,
                "df_html": pd_styler(raw_data)
            }
        return JsonResponse(results)