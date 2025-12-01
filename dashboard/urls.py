from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="dashboard_index"),
    path("weather/<int:system_id>/", views.plot_weather, name="plot_weather"),
    path("features/<int:system_id>/", views.plot_features, name="plot_features"),
    path("load-models/", views.load_models, name="load_models"),
    path("prediction/<int:system_id>/", views.plot_prediction, name="plot_prediction"),
]
