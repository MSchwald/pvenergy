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
    path("weather/<int:system_id>/", views.plot_weather, name="plot_weather"),
    path("features/<int:system_id>/", views.plot_features, name="plot_features"),
    path("load-models/", views.load_models, name="load_models"),
    path("prediction/<int:system_id>/", views.plot_prediction, name="plot_prediction"),
]