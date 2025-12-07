from django.urls import path
from . import views

urlpatterns = [
    path("", views.individual_systems, name="dashboard_individual_systems"),
    path("weather/<int:system_id>/", views.plot_weather, name="plot_weather"),
    path("features/<int:system_id>/", views.plot_features, name="plot_features"),
    path("load-models/", views.load_models, name="load_models"),
    path("prediction/<int:system_id>/", views.plot_prediction, name="plot_prediction"),
    
    path("all_systems/", views.all_systems, name="dashboard_all_systems"),
    path("feature_database/", views.feature_database, name="dashboard_feature_database"),
    path("ml_models/", views.models_info, name="dashboard_ml_models"),
]
