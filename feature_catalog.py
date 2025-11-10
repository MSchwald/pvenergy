from dataclasses import dataclass
from typing import Any, Union
import pandas as pd
from enum import Enum
import numpy as np

class Source(Enum):
    """Indicates if a feature gets calculated or requested from an external source"""
    PVDAQ = "pvdaq"
    PVDAQ_META = "pvdaq_meta"
    NSRDB = "nsrdb"
    PVLIB = "pvlib"
    CALCULATED = "calculated"

@dataclass(frozen = True)
class Feature:
    name: str
    source: Source
    data_type: Any = np.float32
    is_constant: bool = False
    unit: str = ""
    def __str__(self):
        return self.name
    def __repr__(self):
        return f"Feature({self.name})"

class FeatureCatalog:
    # PVDAQ features from the measured pv data of pvdaq pv systems
    PVDAQ_DC_POWER = Feature("dc_power_measured", Source.PVDAQ, unit="W")
    PVDAQ_MODULE_TEMP = Feature("module_temp_measured", Source.PVDAQ, unit="°C")
    PVDAQ_POA_IRRADIANCE = Feature("poa_irradiance_measured", Source.PVDAQ, unit="W/m^2")

    # PVDAQ metadata of pvdaq pv systems 
    # Features might be worth modelling as (usually constant) features into ML models
    AREA = Feature("area", Source.PVDAQ_META, is_constant = True, unit="m^2")
    AZIMUTH = Feature("azimuth", Source.PVDAQ_META, is_constant = True, unit="°")
    TILT = Feature("tilt", Source.PVDAQ_META, is_constant = True, unit="°")
    # Features not suited for ML models but needed for calculations
    SYSTEM_ID = Feature("system_id", Source.PVDAQ_META, data_type = int, is_constant = True)
    ELEVATION = Feature("elevation", Source.PVDAQ_META, is_constant = True, unit="m")
    LATITUDE = Feature("latitude", Source.PVDAQ_META, is_constant = True, unit="°")
    LONGITUDE = Feature("longitude", Source.PVDAQ_META, is_constant = True, unit="°")

    # NSRDB features from the measured weather data
    AIR_TEMP = Feature("air_temperature", Source.NSRDB, unit="°C")
    DHI = Feature("dhi", Source.NSRDB, unit="W/m^2")
    DNI = Feature("dni", Source.NSRDB, unit="W/m^2")
    GHI = Feature("ghi", Source.NSRDB, unit="W/m^2")
    SURFACE_ALBEDO = Feature("surface_albedo", Source.NSRDB, unit="W/m^2")
    WIND_SPEED = Feature("wind_speed", Source.NSRDB, unit="m/s")
    
    # CALCULATED features, derived from the above
    # Derived features calculated with pvlib
    SOLAR_ZENITH = Feature("solar_zenith", Source.CALCULATED, unit="°")
    SOLAR_AZIMUTH = Feature("solar_azimuth", Source.CALCULATED, unit="°")
    PVLIB_POA_IRRADIANCE = Feature("poa_irradiance_calculated", Source.CALCULATED, unit="W/m^2")
    AOI = Feature("aoi_calculated", Source.CALCULATED, unit="°")
    # Features of the Faiman model to calculate the module's temperature
    # faiman_module_temperature = air_temperature + (alpha * poa_irridiance) / (u0 + u1 * wind_speed)
    DCP0 = Feature("dcp0", Source.CALCULATED, unit="W")
    GAMMA = Feature("gamma", Source.CALCULATED, unit="1/K")
    FAIMAN_U0 = Feature("faiman_u0", Source.CALCULATED, unit="W/(m^2*K)") # General Heat Loss Coefficient ("Convective Heat Loss Coefficient at Zero Wind Speed")
    FAIMAN_U1 = Feature("faiman_u1", Source.CALCULATED, unit="W*s/(m^3*K)") # Wind-Dependent Heat Loss Coefficient ("Wind Speed Coefficient")
    FAIMAN_MODULE_TEMP = Feature("module_temp_calculated", Source.CALCULATED, unit="°C")
    TEMP_DIFFERENCE_MEASURED = Feature("temp_difference_measured", Source.CALCULATED, unit="K") # difference between measured module and air temperature
 
    # Time features
    # All dataframes get indexed by timestamps read from PVDAQ data
    # Localized time is relevant for pvlib functions
    TIME = Feature("time", Source.PVDAQ, data_type=pd.DatetimeIndex)
    YEAR = Feature("year", Source.CALCULATED, data_type=int)
    HOUR = Feature("hour", Source.CALCULATED, data_type=int)
    TIME_ZONE = Feature("time_zone", Source.CALCULATED, data_type = str, is_constant = True)
    LOCALIZED_TIME = Feature("localized_time", Source.CALCULATED, data_type=pd.DatetimeIndex)
    # Time features to model degradation, seasonal soiling and daily heat inertia 
    DAYS_SINCE_START = Feature("days_since_start", Source.CALCULATED, unit="d")
    DAY_OF_YEAR = Feature("day_of_year", Source.CALCULATED, data_type=int, unit="d")
    TIME_SINCE_SUNRISE = Feature("time_since_sunrise", Source.CALCULATED)

    # Other derived features
    POWER_RATIO = Feature("power_ratio", Source.CALCULATED)
    COS_AOI = Feature("cos_aoi", Source.CALCULATED)
    POA_CLEAR_SKY_RATIO = Feature("poa_clear_sky_ratio", Source.CALCULATED)
    PVLIB_DC_POWER = Feature("dc_power_calculated", Source.CALCULATED, unit="W")

