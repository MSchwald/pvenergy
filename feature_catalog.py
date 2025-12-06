from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pandas as pd
from enum import Enum

class Source(Enum):
    """Indicates if a feature gets calculated or requested from an external source"""
    PVDAQ = "pvdaq"
    PVDAQ_META = "pvdaq_meta"
    NSRDB = "nsrdb"
    CALCULATED = "calculated"

@dataclass(frozen = True)
class Feature:
    name: str
    source: Source = Source.CALCULATED
    data_type: Any = float
    is_constant: bool = False
    unit: str = ""
    required_features: tuple[Feature] = tuple()
    optional_features: tuple[Feature] = tuple()
    label: str = ""
    description: str = ""

    @property
    def display_name(self) -> str:
        if self.label == "":
            return self.name.replace("_", " ").title()
        return self.label

    @property
    def display_name_with_unit(self) -> str:
        if self.unit == "":
            return self.display_name
        return self.display_name + f" [{self.unit}]"

    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"Feature({self.name})"

class FeatureCatalog:
    """
    Features to be considered to analyze the data of PVDAQ pv systems.
    They are listed in a logical order of their requirements and by themes.
    """
    # Metadata of PVDAQ pv systems
    # The system id is used as the identifier to determine all the others, when available
    SYSTEM_ID = Feature("system_id", Source.PVDAQ_META, is_constant = True, data_type = int)
    # Features that might be worth modelling as (usually constant) features into ML models
    AREA = Feature("area", Source.PVDAQ_META, is_constant = True, unit="m²", required_features=(SYSTEM_ID,))
    AZIMUTH = Feature("azimuth", Source.PVDAQ_META, is_constant = True, unit="°", required_features=(SYSTEM_ID,))
    TILT = Feature("tilt", Source.PVDAQ_META, is_constant = True, unit="°", required_features=(SYSTEM_ID,))
    # Geographical features, only relevant for requesting weather data
    ELEVATION = Feature("elevation", Source.PVDAQ_META, is_constant = True, unit="m", required_features=(SYSTEM_ID,))
    LATITUDE = Feature("latitude", Source.PVDAQ_META, is_constant = True, unit="°N", required_features=(SYSTEM_ID,))
    LONGITUDE = Feature("longitude", Source.PVDAQ_META, is_constant = True, unit="°E", required_features=(SYSTEM_ID,))

    # PVDAQ features from the measured pv data of pvdaq pv systems
    PVDAQ_DC_POWER = Feature("dc_power_measured", Source.PVDAQ, unit="W", required_features=(SYSTEM_ID,))
    PVDAQ_MODULE_TEMP = Feature("module_temp_measured", Source.PVDAQ, unit="°C", required_features=(SYSTEM_ID,))
    PVDAQ_POA_IRRADIANCE = Feature("poa_irradiance_measured", Source.PVDAQ, unit="W/m²", required_features=(SYSTEM_ID,), label="POA Irradiance Measured")

    # Time features
    # All dataframes get indexed by time series read from PVDAQ data. Other sources for TIME could also be allowed.
    # Localized time is relevant for pvlib functions, utc time for OpenMeteo weather forecasts and comparing systems
    UTC_TIME = Feature("utc_time", data_type=pd.DatetimeIndex)
    TIME = Feature("time", Source.PVDAQ, data_type=pd.DatetimeIndex, required_features=(SYSTEM_ID,), label="Local Time")
    YEAR = Feature("year", data_type=int, required_features=(TIME,))
    DAY = Feature("day", data_type=int, required_features=(TIME,))
    HOUR = Feature("hour", data_type=int,  required_features=(TIME,))
    TIME_ZONE = Feature("time_zone", data_type = str, is_constant = True, required_features=(LATITUDE, LONGITUDE))
    UTC_OFFSET = Feature("utc_offset", unit="h", data_type = int, is_constant = True, required_features=(TIME_ZONE,), label="UTC Offset")
    LOCALIZED_TIME = Feature("localized_time", unit="h", data_type=pd.DatetimeIndex, required_features=(TIME, TIME_ZONE))
    # Time features to model degradation, seasonal soiling and daily heat inertia 
    DAYS_SINCE_START = Feature("days_since_start", unit="d", required_features=(TIME,))
    DAY_OF_YEAR = Feature("day_of_year", data_type=int, unit="d", required_features=(TIME,))
    ANNUAL_COSINUS = Feature("annual_sinus", unit="", required_features=(DAY_OF_YEAR,))

    # NSRDB features from measured weather data
    AIR_TEMP = Feature("air_temperature", Source.NSRDB, unit="°C", required_features=(TIME, LATITUDE, LONGITUDE))
    NSRDB_CLEAR_SKY_DHI = Feature("Nsrdb_clear_sky_dhi", Source.NSRDB, unit="W/m²", required_features=(TIME, LATITUDE, LONGITUDE))
    NSRDB_CLEAR_SKY_DNI = Feature("Nsrdb_clear_sky_dni", Source.NSRDB, unit="W/m²", required_features=(TIME, LATITUDE, LONGITUDE))
    NSRDB_CLEAR_SKY_GHI = Feature("Nsrdb_clear_sky_ghi", Source.NSRDB, unit="W/m²", required_features=(TIME, LATITUDE, LONGITUDE))
    DHI = Feature("dhi", Source.NSRDB, unit="W/m²", required_features=(TIME, LATITUDE, LONGITUDE), label="DHI")
    DNI = Feature("dni", Source.NSRDB, unit="W/m²", required_features=(TIME, LATITUDE, LONGITUDE), label="DNI")
    GHI = Feature("ghi", Source.NSRDB, unit="W/m²y", required_features=(TIME, LATITUDE, LONGITUDE), label="GHI")
    SURFACE_ALBEDO = Feature("surface_albedo", Source.NSRDB, unit="W/m²", required_features=(TIME, LATITUDE, LONGITUDE))
    WIND_SPEED = Feature("wind_speed", Source.NSRDB, unit="m/s", required_features=(TIME, LATITUDE, LONGITUDE))
    WIND_DIRECTION = Feature("wind_direction", Source.NSRDB, unit="°", required_features=(TIME, LATITUDE, LONGITUDE))    

    # CALCULATED features, derived from the above
    # Derived features calculated with pvlib
    SOLAR_ZENITH = Feature("solar_zenith", unit="°", required_features=(LOCALIZED_TIME, LATITUDE, LONGITUDE, ELEVATION))
    SOLAR_AZIMUTH = Feature("solar_azimuth", unit="°", required_features=(LOCALIZED_TIME, LATITUDE, LONGITUDE, ELEVATION))
    SOLAR_UNCORRECTED_ZENITH = Feature("solar_uncorrected_zenith", unit="°", required_features=(LOCALIZED_TIME, LATITUDE, LONGITUDE, ELEVATION))
    SOLAR_ELEVATION = Feature("solar_elevantion", unit="°", required_features=(LOCALIZED_TIME, LATITUDE, LONGITUDE, ELEVATION))
    PVLIB_POA_IRRADIANCE = Feature("poa_irradiance_calculated", unit="W/m²", required_features=(TILT, AZIMUTH, SOLAR_ZENITH, SOLAR_AZIMUTH, DNI, GHI, DHI), optional_features=(SURFACE_ALBEDO,), label="POA Irradiance")
    AOI = Feature("aoi_calculated", unit="°", required_features=(TILT, AZIMUTH, SOLAR_ZENITH, SOLAR_AZIMUTH), label="Solar AOI")
    # Features of the Faiman model to calculate the module's temperature; if no module temperature is measured, model uses pvlib default values for u0, u1
    # faiman_module_temperature = air_temperature + (alpha * poa_irridiance) / (u0 + u1 * wind_speed)
    TEMP_DIFFERENCE_MEASURED = Feature("temp_difference_measured", unit="K", required_features=(AIR_TEMP, PVDAQ_MODULE_TEMP)) # difference between measured module and air temperature
    FAIMAN_U0 = Feature("faiman_u0", unit="W/m²K", is_constant = True, optional_features=(WIND_SPEED, PVLIB_POA_IRRADIANCE, TEMP_DIFFERENCE_MEASURED)) # General Heat Loss Coefficient ("Convective Heat Loss Coefficient at Zero Wind Speed")
    FAIMAN_U1 = Feature("faiman_u1", unit="W*s/m³K", is_constant = True, optional_features=(WIND_SPEED, PVLIB_POA_IRRADIANCE, TEMP_DIFFERENCE_MEASURED)) # Wind-Dependent Heat Loss Coefficient ("Wind Speed Coefficient")
    FAIMAN_MODULE_TEMP = Feature("module_temp_calculated", unit="°C", required_features=(PVLIB_POA_IRRADIANCE, WIND_SPEED, FAIMAN_U0, FAIMAN_U1))
    DCP0 = Feature("dcp0", unit="W", is_constant = True, required_features=(YEAR, FAIMAN_MODULE_TEMP, PVLIB_POA_IRRADIANCE, PVDAQ_DC_POWER), optional_features=(AREA,), label="DCP0")
    GAMMA = Feature("gamma", unit="1/K", is_constant = True, required_features=(YEAR, FAIMAN_MODULE_TEMP, PVLIB_POA_IRRADIANCE, PVDAQ_DC_POWER), optional_features=(AREA,))
    # DC_POWER = POA / 1000 * DCP0 * (1 + GAMMA * (TEMP_DIFFERENCE))
    PVLIB_DC_POWER = Feature("dc_power_calculated", unit="W", required_features=(PVLIB_POA_IRRADIANCE, DCP0, GAMMA, FAIMAN_MODULE_TEMP))

    # Location.get_clearsky(times, model='ineichen', solar_position=None, dni_extra=None, **kwargs)
    PVLIB_CLEAR_SKY_GHI = Feature("pvlib_clear_sky_ghi", required_features=(LATITUDE, LONGITUDE, TIME_ZONE, ELEVATION, TIME, SOLAR_ZENITH, SOLAR_UNCORRECTED_ZENITH, SOLAR_ELEVATION), label="Pvlib Clear Sky GHI")
    PVLIB_CLEAR_SKY_DHI = Feature("pvlib_clear_sky_dhi", required_features=(LATITUDE, LONGITUDE, TIME_ZONE, ELEVATION, TIME, SOLAR_ZENITH, SOLAR_UNCORRECTED_ZENITH, SOLAR_ELEVATION), label="Pvlib Clear Sky DHI")
    PVLIB_CLEAR_SKY_DNI = Feature("pvlib_clear_sky_dni", required_features=(LATITUDE, LONGITUDE, TIME_ZONE, ELEVATION, TIME, SOLAR_ZENITH, SOLAR_UNCORRECTED_ZENITH, SOLAR_ELEVATION), label="Pvlib Clear Sky DNI")
    PVLIB_CLEAR_SKY_POA = Feature("pvlib_clear_sky_poa", required_features=(LATITUDE, LONGITUDE, TIME_ZONE, ELEVATION, PVLIB_CLEAR_SKY_GHI, PVLIB_CLEAR_SKY_DHI, PVLIB_CLEAR_SKY_DNI, TILT, AZIMUTH), label="Pvlib Clear Sky POA")
    PVLIB_CLEAR_SKY_RATIO = Feature("pvlib_clear_sky_ratio", required_features=(PVLIB_POA_IRRADIANCE, PVLIB_CLEAR_SKY_POA))
    NSRDB_CLEAR_SKY_POA = Feature("Nsrdb_clear_sky_poa", required_features=(LATITUDE, LONGITUDE, TIME_ZONE, ELEVATION, NSRDB_CLEAR_SKY_GHI, NSRDB_CLEAR_SKY_DHI, NSRDB_CLEAR_SKY_DNI, TILT, AZIMUTH), label="Nsrdb Clear Sky POA")
    NSRDB_CLEAR_SKY_RATIO = Feature("Nsrdb_clear_sky_ratio", required_features=(PVLIB_POA_IRRADIANCE, NSRDB_CLEAR_SKY_POA))
    # The following features use Nsrdb data if available, otherwise they use pvlib estimates as fallback options
    CLEAR_SKY_POA = Feature("clear_sky_poa", required_features=(LATITUDE, LONGITUDE, TIME_ZONE, ELEVATION, TILT, AZIMUTH), optional_features=(NSRDB_CLEAR_SKY_GHI, NSRDB_CLEAR_SKY_DHI, NSRDB_CLEAR_SKY_DNI), label="Clear Sky POA")
    CLEAR_SKY_RATIO = Feature("clear_sky_ratio", required_features=(PVLIB_POA_IRRADIANCE, CLEAR_SKY_POA))


    # Time feature to model thermal inertia
    TIME_SINCE_SUNLIGHT = Feature("time_since_sunrise", unit="h", required_features=(TIME, DAY, PVLIB_POA_IRRADIANCE))

    # Other composed features with non-linear influence on the model
    POWER_RATIO = Feature("power_ratio", required_features=(PVLIB_DC_POWER, DCP0))
    COS_AOI = Feature("cos(aoi)", required_features=(AOI,), label="cos(AOI)")
    RELATIVE_WIND_DIRECTION = Feature("relative_wind_direction", unit="°", required_features=(AZIMUTH, WIND_DIRECTION))
    WIND_NORMAL_COMPONENT = Feature("wind_normal_component", unit="m/s", required_features=(RELATIVE_WIND_DIRECTION, WIND_SPEED, TILT))

    POA_COS_AOI = Feature("poa_*_cos(aoi)", unit="W/m²", required_features = (PVLIB_POA_IRRADIANCE, COS_AOI), label="POA * cos(AOI)")
    #POA_PER_MODULE_TEMP = Feature("poa/module_temp", required_features = (PVLIB_POA_IRRADIANCE, FAIMAN_MODULE_TEMP))
    POA_WIND_SPEED = Feature("poa_*_wind_speed", unit="W/sm", required_features = (PVLIB_POA_IRRADIANCE, WIND_SPEED), label="POA * Wind Speed")
    DCP_PER_AREA = Feature("dcp_/_area", unit="W/m²", required_features = (PVLIB_DC_POWER, AREA), label="DCP / Area")
    DHI_PER_GHI = Feature("dhi_/_ghi", required_features = (DHI, GHI), label="DHI / GHI")
    GAMMA_TEMP_DIFFERENCE = Feature("gamma_*_temp_diff", required_features = (GAMMA, AIR_TEMP, FAIMAN_MODULE_TEMP))
    GAMMA_POA = Feature("gamma_*_poa", unit="W/m²K", required_features = (GAMMA, PVLIB_POA_IRRADIANCE), label="Gamma / POA")
    RELATIVE_AZIMUTH = Feature("relative_azimuth", unit="°", required_features = (AZIMUTH, SOLAR_AZIMUTH))


    