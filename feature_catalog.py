from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pandas as pd
from enum import Enum
import numpy as np

class Source(Enum):
    """Indicates if a feature gets calculated or requested from an external source"""
    PVDAQ = "pvdaq"
    PVDAQ_META = "pvdaq_meta"
    NSRDB = "nsrdb"
    CALCULATED = "calculated"

@dataclass(frozen = True)
class Feature:
    name: str
    source: Source
    data_type: Any = np.float32
    is_constant: bool = False
    unit: str = ""
    required_features: tuple[Feature] = tuple()
    optional_features: tuple[Feature] = tuple()
    def __str__(self):
        return self.name
    def __repr__(self):
        return f"Feature({self.name})"

class FeatureCatalog:
    """
    Features to be considered to analyze the data of PVDAQ pv systems.
    They are listed in a logical order of their requirements and by themes.
    """
    # Metadata of PVDAQ pv systems
    # The system id is used as the identifier to determine all the others, when available
    SYSTEM_ID = Feature("system_id", Source.PVDAQ_META, data_type = int, is_constant = True)
    # Features that might be worth modelling as (usually constant) features into ML models
    AREA = Feature("area", Source.PVDAQ_META, is_constant = True, unit="m^2", required_features=(SYSTEM_ID,))
    AZIMUTH = Feature("azimuth", Source.PVDAQ_META, is_constant = True, unit="°", required_features=(SYSTEM_ID,))
    TILT = Feature("tilt", Source.PVDAQ_META, is_constant = True, unit="°", required_features=(SYSTEM_ID,))
    # Geographical features, only relevant for requesting weather data
    ELEVATION = Feature("elevation", Source.PVDAQ_META, is_constant = True, unit="m", required_features=(SYSTEM_ID,))
    LATITUDE = Feature("latitude", Source.PVDAQ_META, is_constant = True, unit="°", required_features=(SYSTEM_ID,))
    LONGITUDE = Feature("longitude", Source.PVDAQ_META, is_constant = True, unit="°", required_features=(SYSTEM_ID,))

    # PVDAQ features from the measured pv data of pvdaq pv systems
    PVDAQ_DC_POWER = Feature("dc_power_measured", Source.PVDAQ, unit="W", required_features=(SYSTEM_ID,))
    PVDAQ_MODULE_TEMP = Feature("module_temp_measured", Source.PVDAQ, unit="°C", required_features=(SYSTEM_ID,))
    PVDAQ_POA_IRRADIANCE = Feature("poa_irradiance_measured", Source.PVDAQ, unit="W/m^2", required_features=(SYSTEM_ID,))

    # Time features
    # All dataframes get indexed by time series read from PVDAQ data. Other sources for TIME could also be allowed.
    # Localized time is relevant for pvlib functions
    TIME = Feature("time", Source.PVDAQ, data_type=pd.DatetimeIndex, required_features=(SYSTEM_ID,))
    YEAR = Feature("year", Source.CALCULATED, data_type=int, required_features=(TIME,))
    DAY = Feature("day", Source.CALCULATED, data_type=int, required_features=(TIME,))
    HOUR = Feature("hour", Source.CALCULATED, data_type=int,  required_features=(TIME,))
    TIME_ZONE = Feature("time_zone", Source.CALCULATED, data_type = str, is_constant = True, required_features=(LATITUDE, LONGITUDE))
    LOCALIZED_TIME = Feature("localized_time", Source.CALCULATED, data_type=pd.DatetimeIndex, required_features=(TIME, TIME_ZONE))
    # Time features to model degradation, seasonal soiling and daily heat inertia 
    DAYS_SINCE_START = Feature("days_since_start", Source.CALCULATED, unit="d", required_features=(TIME,))
    DAY_OF_YEAR = Feature("day_of_year", Source.CALCULATED, data_type=int, unit="d", required_features=(TIME,))
    ANNUAL_COSINUS = Feature("annual_sinus", Source.CALCULATED, data_type=float, unit="", required_features=(DAY_OF_YEAR,))

    # NSRDB features from measured weather data
    AIR_TEMP = Feature("air_temperature", Source.NSRDB, unit="°C", required_features=(TIME, LATITUDE, LONGITUDE))
    NSRDB_CLEAR_SKY_DHI = Feature("Nsrdb_clear_sky_dhi", Source.NSRDB, unit="W/m^2", required_features=(TIME, LATITUDE, LONGITUDE))
    NSRDB_CLEAR_SKY_DNI = Feature("Nsrdb_clear_sky_dni", Source.NSRDB, unit="W/m^2", required_features=(TIME, LATITUDE, LONGITUDE))
    NSRDB_CLEAR_SKY_GHI = Feature("Nsrdb_clear_sky_ghi", Source.NSRDB, unit="W/m^2", required_features=(TIME, LATITUDE, LONGITUDE))
    DHI = Feature("dhi", Source.NSRDB, unit="W/m^2", required_features=(TIME, LATITUDE, LONGITUDE))
    DNI = Feature("dni", Source.NSRDB, unit="W/m^2", required_features=(TIME, LATITUDE, LONGITUDE))
    GHI = Feature("ghi", Source.NSRDB, unit="W/m^2", required_features=(TIME, LATITUDE, LONGITUDE))
    SURFACE_ALBEDO = Feature("surface_albedo", Source.NSRDB, unit="W/m^2", required_features=(TIME, LATITUDE, LONGITUDE))
    WIND_SPEED = Feature("wind_speed", Source.NSRDB, unit="m/s", required_features=(TIME, LATITUDE, LONGITUDE))
    WIND_DIRECTION = Feature("wind_direction", Source.NSRDB, unit="°", required_features=(TIME, LATITUDE, LONGITUDE))    

    # CALCULATED features, derived from the above
    # Derived features calculated with pvlib
    SOLAR_ZENITH = Feature("solar_zenith", Source.CALCULATED, unit="°", required_features=(LOCALIZED_TIME, LATITUDE, LONGITUDE, ELEVATION))
    SOLAR_AZIMUTH = Feature("solar_azimuth", Source.CALCULATED, unit="°", required_features=(LOCALIZED_TIME, LATITUDE, LONGITUDE, ELEVATION))
    SOLAR_UNCORRECTED_ZENITH = Feature("solar_uncorrected_zenith", Source.CALCULATED, unit="°", required_features=(LOCALIZED_TIME, LATITUDE, LONGITUDE, ELEVATION))
    SOLAR_ELEVATION = Feature("solar_elevantion", Source.CALCULATED, unit="°", required_features=(LOCALIZED_TIME, LATITUDE, LONGITUDE, ELEVATION))
    PVLIB_POA_IRRADIANCE = Feature("poa_irradiance_calculated", Source.CALCULATED, unit="W/m^2", required_features=(TILT, AZIMUTH, SOLAR_ZENITH, SOLAR_AZIMUTH, DNI, GHI, DHI, SURFACE_ALBEDO))
    AOI = Feature("aoi_calculated", Source.CALCULATED, unit="°", required_features=(TILT, AZIMUTH, SOLAR_ZENITH, SOLAR_AZIMUTH))
    # Features of the Faiman model to calculate the module's temperature; if no module temperature is measured, model uses pvlib default values for u0, u1
    # faiman_module_temperature = air_temperature + (alpha * poa_irridiance) / (u0 + u1 * wind_speed)
    TEMP_DIFFERENCE_MEASURED = Feature("temp_difference_measured", Source.CALCULATED, unit="K", required_features=(AIR_TEMP, PVDAQ_MODULE_TEMP)) # difference between measured module and air temperature
    FAIMAN_U0 = Feature("faiman_u0", Source.CALCULATED, unit="W/(m^2*K)", optional_features=(WIND_SPEED, PVLIB_POA_IRRADIANCE, TEMP_DIFFERENCE_MEASURED)) # General Heat Loss Coefficient ("Convective Heat Loss Coefficient at Zero Wind Speed")
    FAIMAN_U1 = Feature("faiman_u1", Source.CALCULATED, unit="W*s/(m^3*K)", optional_features=(WIND_SPEED, PVLIB_POA_IRRADIANCE, TEMP_DIFFERENCE_MEASURED)) # Wind-Dependent Heat Loss Coefficient ("Wind Speed Coefficient")
    FAIMAN_MODULE_TEMP = Feature("module_temp_calculated", Source.CALCULATED, unit="°C", required_features=(PVLIB_POA_IRRADIANCE, WIND_SPEED, FAIMAN_U0, FAIMAN_U1))
    DCP0 = Feature("dcp0", Source.CALCULATED, unit="W", required_features=(YEAR, FAIMAN_MODULE_TEMP, PVLIB_POA_IRRADIANCE, PVDAQ_DC_POWER), optional_features=(AREA,))
    GAMMA = Feature("gamma", Source.CALCULATED, unit="1/K", required_features=(YEAR, FAIMAN_MODULE_TEMP, PVLIB_POA_IRRADIANCE, PVDAQ_DC_POWER), optional_features=(AREA,))
    # DC_POWER = POA / 1000 * DCP0 * (1 + GAMMA * (TEMP_DIFFERENCE))
    PVLIB_DC_POWER = Feature("dc_power_calculated", Source.CALCULATED, unit="W", required_features=(PVLIB_POA_IRRADIANCE, DCP0, GAMMA, FAIMAN_MODULE_TEMP))

    # Location.get_clearsky(times, model='ineichen', solar_position=None, dni_extra=None, **kwargs)
    PVLIB_CLEAR_SKY_GHI = Feature("Pvlib_clear_sky_ghi", Source.CALCULATED, required_features=(LATITUDE, LONGITUDE, TIME_ZONE, ELEVATION, TIME, SOLAR_ZENITH, SOLAR_UNCORRECTED_ZENITH, SOLAR_ELEVATION))
    PVLIB_CLEAR_SKY_DHI = Feature("Pvlib_clear_sky_dhi", Source.CALCULATED, required_features=(LATITUDE, LONGITUDE, TIME_ZONE, ELEVATION, TIME, SOLAR_ZENITH, SOLAR_UNCORRECTED_ZENITH, SOLAR_ELEVATION))
    PVLIB_CLEAR_SKY_DNI = Feature("Pvlib_clear_sky_dni", Source.CALCULATED, required_features=(LATITUDE, LONGITUDE, TIME_ZONE, ELEVATION, TIME, SOLAR_ZENITH, SOLAR_UNCORRECTED_ZENITH, SOLAR_ELEVATION))
    PVLIB_CLEAR_SKY_POA = Feature("Pvlib_clear_sky_poa", Source.CALCULATED, required_features=(LATITUDE, LONGITUDE, TIME_ZONE, ELEVATION, PVLIB_CLEAR_SKY_GHI, PVLIB_CLEAR_SKY_DHI, PVLIB_CLEAR_SKY_DNI, TILT, AZIMUTH))
    PVLIB_CLEAR_SKY_RATIO = Feature("Pvlib_clear_sky_ratio", Source.CALCULATED, required_features=(PVLIB_POA_IRRADIANCE, PVLIB_CLEAR_SKY_POA))
    NSRDB_CLEAR_SKY_POA = Feature("Nsrdb_clear_sky_poa", Source.CALCULATED, required_features=(LATITUDE, LONGITUDE, TIME_ZONE, ELEVATION, NSRDB_CLEAR_SKY_GHI, NSRDB_CLEAR_SKY_DHI, NSRDB_CLEAR_SKY_DNI, TILT, AZIMUTH))
    NSRDB_CLEAR_SKY_RATIO = Feature("Nsrdb_clear_sky_ratio", Source.CALCULATED, required_features=(PVLIB_POA_IRRADIANCE, NSRDB_CLEAR_SKY_POA))

    # Time feature to model thermal inertia
    TIME_SINCE_SUNLIGHT = Feature("time_since_sunrise", Source.CALCULATED, required_features=(TIME, DAY, PVLIB_POA_IRRADIANCE))

    # Other composed features with non-linear influence on the model
    POWER_RATIO = Feature("power_ratio", Source.CALCULATED, required_features=(PVLIB_DC_POWER, DCP0))
    COS_AOI = Feature("cos(aoi)", Source.CALCULATED, required_features=(AOI,))
    RELATIVE_WIND_DIRECTION = Feature("relative_wind_direction", Source.CALCULATED, unit="°", required_features=(AZIMUTH, WIND_DIRECTION))
    WIND_NORMAL_COMPONENT = Feature("wind_normal_component", Source.CALCULATED, unit="m/s", required_features=(RELATIVE_WIND_DIRECTION, WIND_SPEED, TILT))

    POA_COS_AOI = Feature("poa*cos(aoi)", Source.CALCULATED, required_features = (PVLIB_POA_IRRADIANCE, COS_AOI))
    #POA_PER_MODULE_TEMP = Feature("poa/module_temp", Source.CALCULATED, required_features = (PVLIB_POA_IRRADIANCE, FAIMAN_MODULE_TEMP))
    POA_WIND_SPEED = Feature("poa*wind_speed", Source.CALCULATED, required_features = (PVLIB_POA_IRRADIANCE, WIND_SPEED))
    DCP_PER_AREA = Feature("dcp/area", Source.CALCULATED, required_features = (PVLIB_DC_POWER, AREA))
    DHI_PER_GHI = Feature("dhi/ghi", Source.CALCULATED, required_features = (DHI, GHI))
    GAMMA_TEMP_DIFFERENCE = Feature("gamma*temp_diff", Source.CALCULATED, required_features = (GAMMA, AIR_TEMP, FAIMAN_MODULE_TEMP))
    GAMMA_POA = Feature("gamma*poa", Source.CALCULATED, required_features = (GAMMA, PVLIB_POA_IRRADIANCE))
    RELATIVE_AZIMUTH = Feature("relative_azimuth", Source.CALCULATED, required_features = (AZIMUTH, SOLAR_AZIMUTH))


    