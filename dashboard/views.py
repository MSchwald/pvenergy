import pandas as pd
import json

from dataanalysis import Pipeline, Model, ML_MODELS
from data_request import OpenMeteo, Pvdaq
import file_utilities as fu
from feature_catalog import FeatureCatalog as F
from feature_catalog import Feature
from feature_processing import FeatureProcessing as fp
from plotting import Plot
from .formating import feature_format, pd_styler, static_url

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

weather_cache = {}
cache = {}

# Load cached files from the main program
SYSTEM_IDS: tuple[int] = Pipeline.TRAINING_IDS
SYSTEM_CONSTANTS: dict = {id: Pipeline.system_constants(id) for id in SYSTEM_IDS}
LOCATION_FROM_ID = {id : (Pvdaq.meta(id)[F.LATITUDE], Pvdaq.meta(id)[F.LONGITUDE]) for id in SYSTEM_IDS}
LOCATIONS: tuple[int] = tuple({(Pvdaq.meta(id)[F.LATITUDE], Pvdaq.meta(id)[F.LONGITUDE]) for id in SYSTEM_IDS})

# Get loaded while already rendering in order to not delay showing the starting page for several seconds
TRAINED_MODELS = [ML_MODELS.XGBOOST, ML_MODELS.LIGHTGBM, ML_MODELS.RANDOM_FOREST]
ml_models: tuple[Model] = tuple()
training_features: tuple[Feature] = tuple()

class TemplateViews:
    def individual_systems(request):
        return {"ids": SYSTEM_IDS}

    def all_systems(request):
        df = pd.read_csv("results.csv", index_col = 0).map(lambda x: feature_format(x, display_unit = False))
        df.index.name = F.SYSTEM_ID.name
        return {
            "metadata": pd_styler(Pipeline.get_system_constants()),
            "evaluations": pd_styler(df)
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
                } for ftr in fp.ALL_FEATURES
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
        return JsonResponse({
            str(i): {
                    "features": ", ".join([feature_format(name, display_unit = False) for name in model._training_features]),
                    "target": model._target_feature,
                    "evaluations": pd_styler(model._evaluation_results),
                    "parameter": pd_styler(pd.Series(model.get_hyperparameters(), name="Value"))
            } for i, model in enumerate(ml_models)
        })

    def load_metadata(request):
        return JsonResponse({str(id): pd_styler(Pipeline.get_system_constants().loc[id].to_frame().T) for id in SYSTEM_IDS})

    def load_weather(request):
        fetch_locations = [
            (lat, lon) for (lat, lon) in LOCATIONS if not fu.file_up_to_date(
                OpenMeteo.cache_name(lat, lon)
            )
        ]
        for loc in [loc for loc in LOCATIONS if not loc in fetch_locations]:
            if not loc in weather_cache.keys():
                weather_cache[loc] = OpenMeteo.load_cache(OpenMeteo.cache_name(loc[0], loc[1]))
        return JsonResponse({
            "locations": fetch_locations,
            "url": OpenMeteo.url,
            "parameters": OpenMeteo.PARAMETERS
        })

    def load_models(request):
        global ml_models
        global training_features
        if not ml_models:
            ml_models = tuple(Model.load(ml_model.name) for ml_model in TRAINED_MODELS)
            training_features = tuple(fp.FEATURE_FROM_NAME[name] for name in ml_models[0]._training_features)
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
        plots = {}
        for id in SYSTEM_IDS:
            df = weather_cache[LOCATION_FROM_ID[id]].copy()
            df.ftr.set_const(SYSTEM_CONSTANTS[id])
            df = Pipeline.utc_to_local_time(df)
            df.ftr.set_const(SYSTEM_CONSTANTS[id])
            cache[id] = df
            plots[str(id)] = {
                "weather_url": static_url(Plot.weather_forecast(df, id)),
                "weather_title": f"Forecast, starting from {df.index[0]} local time (UTC{df.ftr.get_const(F.UTC_OFFSET)})"
            }
        return JsonResponse(plots)

    def plot_features(request):
        plots = {}
        for id in SYSTEM_IDS:
            df = cache[id]
            plots[str(id)] = {
                "features_plot_url": static_url(Plot.calculated_features(df, id))
            }
        return JsonResponse(plots)

    def plot_predictions(request):
        results = {}
        for id in SYSTEM_IDS:
            df = cache[id]
            Y = Pipeline.predict(ml_models, df)
            energy = pd_styler(pd.Series(data = {ml_model.name: Pipeline.integrate_timeseries(Y[ml_model.name]) for ml_model in ml_models}, name = "Energy [kWh]"))
            raw_data = pd.concat(
                [df.ftr.get(training_features), Y], axis = 1
            )
            results[id] = {
                "prediction_plot_url": static_url(Plot.predict(Y, id)),
                "energy": energy,
                "df_html": pd_styler(raw_data)
            }
        return JsonResponse(results)