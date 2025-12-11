from django.urls import path
from .menu import make_menu_urls
from .views import ApiEndpoints
import inspect

MENU = [
    ("Individual System Forecast", "individual_systems"),
    ("All System Forecast", "all_systems"),
    ("Feature Database", "feature_database"),
    ("Machine Learning Models", "models_info")
]

urlpatterns = make_menu_urls(MENU) + [
    path(f"{name.replace("_","-")}/", getattr(ApiEndpoints, name), name = name)
    for name, method in inspect.getmembers(ApiEndpoints, inspect.isfunction) if not name.startswith("_")
]