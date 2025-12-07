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
    SYSTEM_ID = Feature("system_id", Source.PVDAQ_META, is_constant = True, data_type = int, label="System ID",
        description="Identification number of a PVDAQ photovoltaic system")
    # Features that might be worth modelling as (usually constant) features into ML models
    AREA = Feature("area", Source.PVDAQ_META, is_constant = True, unit="m²", required_features=(SYSTEM_ID,),
        description="Surface area of a photovoltaic module")
    AZIMUTH = Feature("azimuth", Source.PVDAQ_META, is_constant = True, unit="°", required_features=(SYSTEM_ID,),
        description="The compass direction that the photovoltaic module faces; angle measured clockwise from the north")
    TILT = Feature("tilt", Source.PVDAQ_META, is_constant = True, unit="°", required_features=(SYSTEM_ID,),
        description="Tilt angle of the photovoltaic module, measured from the horizontal plane")
    # Geographical features, only relevant for requesting weather data
    ELEVATION = Feature("elevation", Source.PVDAQ_META, is_constant = True, unit="m", required_features=(SYSTEM_ID,),
        description="Elevation of the sytem's location above sea level")
    LATITUDE = Feature("latitude", Source.PVDAQ_META, is_constant = True, unit="°N", required_features=(SYSTEM_ID,),
        description="Geographical latitude of the system's location")
    LONGITUDE = Feature("longitude", Source.PVDAQ_META, is_constant = True, unit="°E", required_features=(SYSTEM_ID,),
        description="Geographical longitude of the system's location")

    # PVDAQ features from the measured pv data of pvdaq pv systems
    PVDAQ_DC_POWER = Feature("dc_power_measured", Source.PVDAQ, unit="W", required_features=(SYSTEM_ID,),
        description="The measured electric power of the direct current produced by the photovoltaic module before conversion by inverters")
    PVDAQ_MODULE_TEMP = Feature("module_temp_measured", Source.PVDAQ, unit="°C", required_features=(SYSTEM_ID,),
        description="The measured temperature of the photovoltaic module")
    PVDAQ_POA_IRRADIANCE = Feature("poa_irradiance_measured", Source.PVDAQ, unit="W/m²", required_features=(SYSTEM_ID,), label="POA Irradiance Measured",
        description="Measured plane-of-array irradiance, the intensity of the total direct and indirect solar radiation hitting the photovoltaic module's surface")

    # Time features
    # All dataframes get indexed by time series read from PVDAQ data. Other sources for TIME could also be allowed.
    UTC_TIME = Feature("utc_time", data_type=pd.DatetimeIndex,
        description="Coordinated Universal Time")
    TIME = Feature("time", Source.PVDAQ, data_type=pd.DatetimeIndex, required_features=(SYSTEM_ID,), label="Local Time",
        description="Local timestamps from PVDAQ measurements (DST-naive)")
    YEAR = Feature("year", data_type=int, required_features=(TIME,))
    DAY = Feature("day", data_type=int, required_features=(TIME,))
    HOUR = Feature("hour", data_type=int,  required_features=(TIME,))
    TIME_ZONE = Feature("time_zone", data_type = str, is_constant = True, required_features=(LATITUDE, LONGITUDE),
        description="Name of the time zone in pytz")
    UTC_OFFSET = Feature("utc_offset", unit="h", data_type = int, is_constant = True, required_features=(TIME_ZONE,), label="UTC Offset",
        description="Difference between local DST-naive time and Coordinated Universal Time (UTC)")
    LOCALIZED_TIME = Feature("localized_time", unit="h", data_type=pd.DatetimeIndex, required_features=(TIME, TIME_ZONE),
        description="Time zone aware time format, used for calculations with pvlib")
    DAYS_SINCE_START = Feature("days_since_start", unit="d", required_features=(TIME,),
        description="Time since first measurement; can be relevant to analyze a system's degradation over time")
    DAY_OF_YEAR = Feature("day_of_year", data_type=int, unit="d", required_features=(TIME,),
        description="The day of the year can be used to model seasonal soiling effects caused by dust, rain etc.")
    ANNUAL_COSINUS = Feature("annual_cosinus", unit="", required_features=(DAY_OF_YEAR,),
        description="A smooth periodic alternative to the day-of-the-year Feature")

    # NSRDB features from measured weather data
    AIR_TEMP = Feature("air_temperature", Source.NSRDB, unit="°C", required_features=(TIME, LATITUDE, LONGITUDE),
        description="Ambient Air Temperature near the photovoltaic module")
    DHI = Feature("dhi", Source.NSRDB, unit="W/m²", required_features=(TIME, LATITUDE, LONGITUDE), label="DHI",
        description="Intensity of solar radiation scattered by the atmosphere onto a horizontal plane")
    DNI = Feature("dni", Source.NSRDB, unit="W/m²", required_features=(TIME, LATITUDE, LONGITUDE), label="DNI",
        description="Intensity of the direct solar radiation onto a plane perpendicular to the sun's angle")
    GHI = Feature("ghi", Source.NSRDB, unit="W/m²", required_features=(TIME, LATITUDE, LONGITUDE), label="GHI",
        description="Total intensity of the direct and scattered solar radiation onto a horizontal plane")
    NSRDB_CLEAR_SKY_DHI = Feature("Nsrdb_clear_sky_dhi", Source.NSRDB, unit="W/m²", required_features=(TIME, LATITUDE, LONGITUDE),
        description="DHI under clear sky conditions, calculated using Nsrdb satellite measurements")
    NSRDB_CLEAR_SKY_DNI = Feature("Nsrdb_clear_sky_dni", Source.NSRDB, unit="W/m²", required_features=(TIME, LATITUDE, LONGITUDE),
        description="DNI under clear sky conditions, calculated using Nsrdb satellite measurement")
    NSRDB_CLEAR_SKY_GHI = Feature("Nsrdb_clear_sky_ghi", Source.NSRDB, unit="W/m²", required_features=(TIME, LATITUDE, LONGITUDE),
        description="GHI under clear sky conditions, calculated using Nsrdb satellite measurements")
    SURFACE_ALBEDO = Feature("surface_albedo", Source.NSRDB, required_features=(TIME, LATITUDE, LONGITUDE),
        description="Fraction of the solar radiation reflected by the ground")
    WIND_SPEED = Feature("wind_speed", Source.NSRDB, unit="m/s", required_features=(TIME, LATITUDE, LONGITUDE),
        description="Wind speed near the photovoltaic module")
    WIND_DIRECTION = Feature("wind_direction", Source.NSRDB, unit="°", required_features=(TIME, LATITUDE, LONGITUDE),
        description="The angle between north and the compass direction the wind is coming from")

    # CALCULATED features, derived from the above
    # Derived features calculated with pvlib
    SOLAR_ZENITH = Feature("solar_zenith", unit="°", required_features=(LOCALIZED_TIME, LATITUDE, LONGITUDE, ELEVATION),
        description="Apparent angle between sun and the zenith, including refraction by the atmosphere")
    SOLAR_AZIMUTH = Feature("solar_azimuth", unit="°", required_features=(LOCALIZED_TIME, LATITUDE, LONGITUDE, ELEVATION),
        description="The compass direction of the sun; angle measured clockwise from the north")
    SOLAR_UNCORRECTED_ZENITH = Feature("solar_uncorrected_zenith", unit="°", required_features=(LOCALIZED_TIME, LATITUDE, LONGITUDE, ELEVATION),
        description="Angle between sun and the zenith, ignoring refraction by the atmosphere")
    SOLAR_ELEVATION = Feature("solar_elevation", unit="°", required_features=(LOCALIZED_TIME, LATITUDE, LONGITUDE, ELEVATION),
        description="Elevation angle of the sun above the horizon")
    PVLIB_POA_IRRADIANCE = Feature("poa_irradiance_calculated", unit="W/m²", required_features=(TILT, AZIMUTH, SOLAR_ZENITH, SOLAR_AZIMUTH, DNI, GHI, DHI), optional_features=(SURFACE_ALBEDO,), label="POA Irradiance",
        description="Plane-of-array irradiance; total solar irradiation hitting the photovoltaic module (estimated using pvlib) from direct, diffuse and reflected components")
    AOI = Feature("aoi_calculated", unit="°", required_features=(TILT, AZIMUTH, SOLAR_ZENITH, SOLAR_AZIMUTH), label="Solar AOI",
        description="Angle of incidence under which the direct sunlight hits onto the photovoltaic module's surface")
    # Features of the Faiman model to calculate the module's temperature; if no module temperature is measured, model uses pvlib default values for u0, u1
    # faiman_module_temperature = air_temperature + (alpha * poa_irridiance) / (u0 + u1 * wind_speed)
    TEMP_DIFFERENCE_MEASURED = Feature("temp_difference_measured", unit="K", required_features=(AIR_TEMP, PVDAQ_MODULE_TEMP),
        description="Difference between the photovoltaic module's temperature and the ambient air temperature")
    FAIMAN_U0 = Feature("faiman_u0", unit="W/m²K", is_constant = True, optional_features=(WIND_SPEED, PVLIB_POA_IRRADIANCE, TEMP_DIFFERENCE_MEASURED),
        description="Convective Heat Loss Coefficient at zero wind speed; describes how fast the photovoltaic module adapts to temperature changes")
    FAIMAN_U1 = Feature("faiman_u1", unit="W*s/m³K", is_constant = True, optional_features=(WIND_SPEED, PVLIB_POA_IRRADIANCE, TEMP_DIFFERENCE_MEASURED),
        description="Wind Speed Coefficient; describes how strongly wind cools the photovoltaic module")
    FAIMAN_MODULE_TEMP = Feature("module_temp_calculated", unit="°C", required_features=(PVLIB_POA_IRRADIANCE, WIND_SPEED, FAIMAN_U0, FAIMAN_U1),
        description="Temperature of the photovoltaic module estimated with the Faiman model")
    DCP0 = Feature("dcp0", unit="W", is_constant = True, required_features=(YEAR, FAIMAN_MODULE_TEMP, PVLIB_POA_IRRADIANCE, PVDAQ_DC_POWER), optional_features=(AREA,), label="DCP0",
        description="Expected DC power under reference conditions (module temperature 25 °C, POA irradiance 1000 W/m²)")
    GAMMA = Feature("gamma", unit="1/K", is_constant = True, required_features=(YEAR, FAIMAN_MODULE_TEMP, PVLIB_POA_IRRADIANCE, PVDAQ_DC_POWER), optional_features=(AREA,),
        description="Coefficient quantifying the relative DC power change per °C deviation from the 25 °C reference")
    # DC_POWER = POA / (1000 W/m²) * DCP0 * (1 + GAMMA * (TEMP_MODULE - 25°C))
    PVLIB_DC_POWER = Feature("dc_power_calculated", unit="W", required_features=(PVLIB_POA_IRRADIANCE, DCP0, GAMMA, FAIMAN_MODULE_TEMP),
        description="DC Power estimated with pvlib using the Faiman model")

    # Location.get_clearsky(times, model='ineichen', solar_position=None, dni_extra=None, **kwargs)
    PVLIB_CLEAR_SKY_GHI = Feature("pvlib_clear_sky_ghi", required_features=(LATITUDE, LONGITUDE, TIME_ZONE, ELEVATION, TIME, SOLAR_ZENITH, SOLAR_UNCORRECTED_ZENITH, SOLAR_ELEVATION), label="Pvlib Clear Sky GHI",
        description="Pvlib estimation of the GHI irradiance under clear sky conditions")
    PVLIB_CLEAR_SKY_DHI = Feature("pvlib_clear_sky_dhi", required_features=(LATITUDE, LONGITUDE, TIME_ZONE, ELEVATION, TIME, SOLAR_ZENITH, SOLAR_UNCORRECTED_ZENITH, SOLAR_ELEVATION), label="Pvlib Clear Sky DHI",
        description="Pvlib estimation of the DHI irradiance under clear sky conditions")
    PVLIB_CLEAR_SKY_DNI = Feature("pvlib_clear_sky_dni", required_features=(LATITUDE, LONGITUDE, TIME_ZONE, ELEVATION, TIME, SOLAR_ZENITH, SOLAR_UNCORRECTED_ZENITH, SOLAR_ELEVATION), label="Pvlib Clear Sky DNI",
        description="Pvlib estimation of the DNI irradiance under clear sky conditions")
    PVLIB_CLEAR_SKY_POA = Feature("pvlib_clear_sky_poa", required_features=(LATITUDE, LONGITUDE, TIME_ZONE, ELEVATION, PVLIB_CLEAR_SKY_GHI, PVLIB_CLEAR_SKY_DHI, PVLIB_CLEAR_SKY_DNI, TILT, AZIMUTH), label="Pvlib Clear Sky POA",
        description="Pvlib estimation of the POA irradiance under clear sky conditions")
    PVLIB_CLEAR_SKY_RATIO = Feature("pvlib_clear_sky_ratio", required_features=(PVLIB_POA_IRRADIANCE, PVLIB_CLEAR_SKY_POA),
        description="Ratio between the POA irradiance calculated from Nsrdb data and the pvlib estimation under clear sky conditions")
    NSRDB_CLEAR_SKY_POA = Feature("Nsrdb_clear_sky_poa", required_features=(LATITUDE, LONGITUDE, TIME_ZONE, ELEVATION, NSRDB_CLEAR_SKY_GHI, NSRDB_CLEAR_SKY_DHI, NSRDB_CLEAR_SKY_DNI, TILT, AZIMUTH), label="Nsrdb Clear Sky POA",
        description="Nsrdb estimation of the POA irradiance under clear sky conditions")
    NSRDB_CLEAR_SKY_RATIO = Feature("Nsrdb_clear_sky_ratio", required_features=(PVLIB_POA_IRRADIANCE, NSRDB_CLEAR_SKY_POA),
        description="Ratio between the POA irradiance calculated from Nsrdb data and the Nsrdb estimation under clear sky conditions")
    CLEAR_SKY_POA = Feature("clear_sky_poa", required_features=(LATITUDE, LONGITUDE, TIME_ZONE, ELEVATION, TILT, AZIMUTH), optional_features=(NSRDB_CLEAR_SKY_GHI, NSRDB_CLEAR_SKY_DHI, NSRDB_CLEAR_SKY_DNI), label="Clear Sky POA",
        description="Estimation of the POA irradiance under clear sky conditions; prefers Nsrdb data over pvlib")
    CLEAR_SKY_RATIO = Feature("clear_sky_ratio", required_features=(PVLIB_POA_IRRADIANCE, CLEAR_SKY_POA),
        description="Ratio between the POA irradiance calculated from Nsrdb data and the estimation under clear sky conditions; prefers Nsrdb data over pvlib")
    
    # Other composed features with non-linear influence on the model
    TIME_SINCE_SUNLIGHT = Feature("time_since_sunrise", unit="h", required_features=(TIME, DAY, PVLIB_POA_IRRADIANCE),
        description="Time since the day's first solar irradiance reached the photovoltaic module; used for modelling thermic inertia of the module")
    POWER_RATIO = Feature("power_ratio", required_features=(PVLIB_DC_POWER, DCP0),
        description="Ratio between DC Power estimated with pvlib and DCP0")
    COS_AOI = Feature("cos(aoi)", required_features=(AOI,), label="cos(AOI)",
        description="The Cosine of the Solar AOI determines the fraction of light that get's reflected by the photovoltaic module's surface")
    RELATIVE_WIND_DIRECTION = Feature("relative_wind_direction", unit="°", required_features=(AZIMUTH, WIND_DIRECTION),
        description="Angle between Wind Direction and the photovoltaic module's Azimuth")
    WIND_NORMAL_COMPONENT = Feature("wind_normal_component", unit="m/s", required_features=(RELATIVE_WIND_DIRECTION, WIND_SPEED, TILT),
        description="Wind component perpendicular to the photovoltaic module's surface")
    POA_COS_AOI = Feature("poa*cos(aoi)", unit="W/m²", required_features = (PVLIB_POA_IRRADIANCE, COS_AOI), label="POA * cos(AOI)",
        description="Product for modelling the total loss of solar irradiation due to reflection on the photovoltaic module's surface")
    POA_WIND_SPEED = Feature("poa*wind_speed", unit="W/m²*m/s", required_features = (PVLIB_POA_IRRADIANCE, WIND_SPEED), label="POA * Wind Speed",
        description="Product to model higher order interactions between solar irradiation and wind")
    DCP_PER_AREA = Feature("dcp/area", unit="W/m²", required_features = (PVLIB_DC_POWER, AREA), label="DCP / Area",
        description="Normalization of estimated DC Power by the photovoltaic module's surface area for training on data of multiple systems")
    DHI_PER_GHI = Feature("dhi/ghi", required_features = (DHI, GHI), label="DHI / GHI",
        description="Fraction of diffuse solar irradiance; good indicator of cloudiness (undefined when GHI is zero)")
    GAMMA_TEMP_DIFFERENCE = Feature("gamma*temp_diff", required_features = (GAMMA, AIR_TEMP, FAIMAN_MODULE_TEMP), label="Gamma * Temp Diff",
        description="Term of the Faiman model quantifying the DC Power loss due to temperature effects")
    GAMMA_POA = Feature("gamma*poa", unit="W/m²K", required_features = (GAMMA, PVLIB_POA_IRRADIANCE), label="Gamma * POA",
        description="Interaction term modeling the combined influence of irradiance and temperature on power losses")
    RELATIVE_AZIMUTH = Feature("relative_azimuth", unit="°", required_features = (AZIMUTH, SOLAR_AZIMUTH),
        description="Difference between the azimuth angles of the photovoltaic module and the sun")