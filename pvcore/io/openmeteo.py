from __future__ import annotations

import requests
import pandas as pd

from pvcore.paths import OPENMETEO_DIR
from pvcore.feature import Catalog as F
from pvcore.utils import file_utilities as fu

class OpenMeteo:
    """Request live / forecast weather data"""
    url = "https://api.open-meteo.com/v1/forecast"
    PARAMETERS = {
            "hourly": [
                "shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance",
                "temperature_2m", "wind_speed_10m", "wind_direction_10m"
            ],
            "models": "gfs_seamless",
            "current_weather": True,
            "forecast_hours": 24
    }
    COLUMN_NAME_MAP = {
        "time": F.UTC_TIME.name,
        "temperature_2m": F.AIR_TEMP.name,
        "shortwave_radiation": F.GHI.name,
        "diffuse_radiation": F.DHI.name,
        "direct_normal_irradiance": F.DNI.name,
        "wind_speed_10m": F.WIND_SPEED.name,
        "wind_direction_10m": F.WIND_DIRECTION.name,
    }
    
    @classmethod
    def cache_name(cls, latitude: float, longitude: float) -> Path:
        return OPENMETEO_DIR / f"openmeteo_lat={latitude}_lon={longitude}.csv"

    @classmethod
    def load_cache(cls, cache: Path) -> pd.DataFrame:
        df = pd.read_csv(cache, index_col = F.UTC_TIME.name)
        df.index = pd.to_datetime(df.index)
        print("Loaded cached version of OpenMeteo data")
        return df

    @classmethod
    def format_response(cls, json: str):
        df = pd.DataFrame(data = json["hourly"]).rename(columns=cls.COLUMN_NAME_MAP).set_index(F.UTC_TIME.name)
        df.index = pd.to_datetime(df.index)
        return df
        
    @classmethod
    def get_forecast(cls,
        latitude: float,
        longitude: float,
        save_result: bool = True
    ) -> pd.DataFrame:
        """
        Get hourly weather forecast for given location for the next day.
        Caching is used for multiple requests in the same hour of the day.
        """
        cache = cls.cache_name(latitude, longitude)
        if fu.file_up_to_date(cache):
            cls.load_cache(cache)
        params = {"latitude": latitude, "longitude": longitude}
        params.update(cls.PARAMETERS)
        print("Request new version from OpenMeteo")
        response = requests.get(
            cls.url,
            params = params,
            timeout = 30
        )
        response.raise_for_status()
        df = cls.format_response(response.json())
        if save_result:
            df.to_csv(cache, index = True)
        return df