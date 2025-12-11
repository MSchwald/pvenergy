from django.urls import path
from . import views
from .menu import make_menu_urls

MENU = [
    ("Individual System Forecast", "individual_systems"),
    ("All System Forecast", "all_systems"),
    ("Feature Database", "feature_database"),
    ("Machine Learning Models", "models_info")
]

urlpatterns = make_menu_urls(MENU) + [
    path("load-metadata/", views.load_metadata, name="load_metadata"),
    path("load-models/", views.load_models, name="load_models"),
    path("load-weather/", views.load_weather, name="load_weather"),
    path("save-weather/", views.save_weather, name="save_weather"),
    path("plot-weather/", views.plot_weather, name="plot_weather"),
    path("plot-features/", views.plot_features, name="plot_features"),
    path("plot-predictions/", views.plot_predictions, name="plot_predictions"),
    path("models-names/", views.models_names, name="models_names"),
    path("models-training-results/", views.models_training_results, name="models_training_results"),
]