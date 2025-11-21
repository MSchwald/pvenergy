from __future__ import annotations
from dataclasses import dataclass
from typing import Union
import pandas as pd
import numpy as np
from feature_catalog import Feature, Source
from feature_catalog import FeatureCatalog as F
from sklearn.linear_model import LinearRegression
import pvlib
import pytz
from timezonefinder import TimezoneFinder
from pathlib import Path

class FeatureProcessing:
    ALL_FEATURES: tuple[Feature] = tuple(
        feature for feature in vars(F).values() if isinstance(feature, Feature)
    )
    ALL_FEATURE_NAMES: tuple[str] = tuple(feature.name for feature in ALL_FEATURES)
    CALCULATED_FEATURES: tuple[Feature] = tuple(
        feature for feature in ALL_FEATURES if feature.source == Source.CALCULATED
    )

    FEATURE_FROM_NAME: dict[str, Feature] = {feature.name: feature for feature in ALL_FEATURES}
    tf = TimezoneFinder()

    @classmethod
    def calculate(cls, feature: Feature, api: FeatureAccessor) -> Union[pd.Series, np.float32]:
        """Defining formulas for calculating derived features."""
        match feature:
            # Features of the Faiman model to calculate the module's temperature
            # faiman_module_temperature = air_temperature + (alpha * poa_irridiance) / (u0 + u1 * wind_speed)
            case F.FAIMAN_MODULE_TEMP:
                return pvlib.temperature.faiman(api.get(F.PVLIB_POA_IRRADIANCE), api.get(F.WIND_SPEED),
                                                api.get_const(F.FAIMAN_U0), api.get_const(F.FAIMAN_U1))
            case F.FAIMAN_U0 | F.FAIMAN_U1:
                u0, u1 = cls.calculate_faiman_coefficients(api)
                api.set_const({F.FAIMAN_U0: u0, F.FAIMAN_U1: u1})
                return api.get_const(feature)
            case F.TEMP_DIFFERENCE_MEASURED:
                return api.get(F.PVDAQ_MODULE_TEMP) - api.get(F.AIR_TEMP)
            case F.DCP0 | F.GAMMA:
                dcp0, gamma = cls.calculate_annual_dcp0_gamma(api)
                api.set_const({F.DCP0: dcp0, F.GAMMA: gamma})
                return api.get_const(feature)
            # Pvlib features (often require localized time)
            case F.SOLAR_ZENITH | F.SOLAR_AZIMUTH | F.SOLAR_UNCORRECTED_ZENITH | F.SOLAR_ELEVATION:
                solpos = pvlib.solarposition.get_solarposition(time = api.get(F.LOCALIZED_TIME),
                                                        latitude = api.get_const(F.LATITUDE),
                                                        longitude = api.get_const(F.LONGITUDE),
                                                        altitude = api.get_const(F.ELEVATION))
                api.set(F.SOLAR_AZIMUTH, solpos["azimuth"])
                api.set(F.SOLAR_ZENITH, solpos["apparent_zenith"])
                api.set(F.SOLAR_UNCORRECTED_ZENITH, solpos["zenith"])
                api.set(F.SOLAR_ELEVATION, solpos["apparent_elevation"])
                return api.get(feature)

            case F.AOI:
                return pvlib.irradiance.aoi(surface_tilt = api.get_const(F.TILT),
                                surface_azimuth = api.get_const(F.AZIMUTH),
                                solar_zenith = api.get(F.SOLAR_ZENITH),
                                solar_azimuth = api.get(F.SOLAR_AZIMUTH))
            case F.PVLIB_DC_POWER:
                return pvlib.pvsystem.pvwatts_dc(api.get(F.PVLIB_POA_IRRADIANCE), 
                                                api.get(F.FAIMAN_MODULE_TEMP),
                                                api.get_const(F.DCP0),
                                                api.get_const(F.GAMMA))
            case F.PVLIB_CLEAR_SKY_GHI | F.PVLIB_CLEAR_SKY_DHI | F.PVLIB_CLEAR_SKY_DNI:
                location = pvlib.location.Location(api.get_const(F.LATITUDE),
                                          api.get_const(F.LONGITUDE),
                                          api.get_const(F.TIME_ZONE),
                                          api.get_const(F.ELEVATION)
                )
                solar_position = api.get([F.SOLAR_ZENITH, F.SOLAR_UNCORRECTED_ZENITH, F.SOLAR_ELEVATION])
                solar_position = solar_position.rename(columns = {F.SOLAR_ZENITH.name: "apparent_zenith",
                                                                F.SOLAR_UNCORRECTED_ZENITH.name: "zenith",
                                                                F.SOLAR_ELEVATION.name: "apparent_elevation"}
                )
                result = location.get_clearsky(pd.DatetimeIndex(api.get(F.TIME)), solar_position = solar_position)
                api.set(F.PVLIB_CLEAR_SKY_GHI, result["ghi"])
                api.set(F.PVLIB_CLEAR_SKY_DHI, result["dhi"])
                api.set(F.PVLIB_CLEAR_SKY_DNI, result["dni"])
                return api.get(feature)
            case F.PVLIB_POA_IRRADIANCE | F.PVLIB_CLEAR_SKY_POA | F.NSRDB_CLEAR_SKY_POA:
                if feature == F.PVLIB_POA_IRRADIANCE:
                    dni, ghi, dhi, albedo = api.get(F.DNI), api.get(F.GHI), api.get(F.DHI), api.get(F.SURFACE_ALBEDO)
                elif feature == F.PVLIB_CLEAR_SKY_POA:
                    dni, ghi, dhi, albedo = api.get(F.PVLIB_CLEAR_SKY_DNI), api.get(F.PVLIB_CLEAR_SKY_GHI), api.get(F.PVLIB_CLEAR_SKY_DHI), 0.25
                else:
                    dni, ghi, dhi, albedo = api.get(F.PVLIB_CLEAR_SKY_DNI), api.get(F.PVLIB_CLEAR_SKY_GHI), api.get(F.PVLIB_CLEAR_SKY_DHI), 0.25
                poa = pvlib.irradiance.get_total_irradiance(surface_tilt = api.get_const(F.TILT),
                                                            surface_azimuth = api.get_const(F.AZIMUTH),
                                                            solar_zenith = api.get(F.SOLAR_ZENITH),
                                                            solar_azimuth = api.get(F.SOLAR_AZIMUTH),
                                                            dni = dni, ghi = ghi, dhi = dhi, albedo = albedo)
                return poa["poa_global"]
            case F.NSRDB_CLEAR_SKY_RATIO | F.PVLIB_CLEAR_SKY_RATIO:
                if feature == F.PVLIB_CLEAR_SKY_RATIO:
                    clear_sky_poa = F.PVLIB_CLEAR_SKY_POA
                else:
                    clear_sky_poa = F.NSRDB_CLEAR_SKY_POA
                df = api.get([F.PVLIB_POA_IRRADIANCE, clear_sky_poa])
                sunny = df[clear_sky_poa.name] >= 1
                ratio = pd.Series(np.nan, index=df.index, dtype=float)
                ratio[sunny] = df.loc[sunny, F.PVLIB_POA_IRRADIANCE.name] / df.loc[sunny, clear_sky_poa.name]
                return ratio.clip(upper = 1)
                
            # Time features
            # General features
            case F.TIME_ZONE:
                tz = cls.tf.timezone_at(lat = api.get_const(F.LATITUDE),
                                          lng = api.get_const(F.LONGITUDE))    
                return tz    
            case F.LOCALIZED_TIME:
                tz = pytz.timezone(api.get_const(F.TIME_ZONE))
                time_series = pd.to_datetime(api.get(F.TIME))
                return time_series.apply(lambda t: tz.localize(t, is_dst = False))
            case F.YEAR:
                return api.get(F.TIME).dt.year
            case F.DAY:
                return api.get(F.TIME).dt.day
            case F.HOUR:
                return api.get(F.TIME).dt.hour
            # Time features to model degradation, seasonal soiling and daily heat inertia
            case F.DAYS_SINCE_START:
                starting_day = api.get(F.TIME).iloc[0]
                return (api.get(F.TIME) - starting_day).dt.days
            case F.DAY_OF_YEAR:
                return api.get(F.TIME).dt.dayofyear
            case F.ANNUAL_COSINUS:
                t = api.get(F.TIME)
                year_start = t.dt.to_period("Y").dt.start_time

                seconds = (t - year_start).dt.total_seconds()
                days_in_year = ((t.dt.to_period("Y").dt.end_time - year_start).dt.days + 1)

                year_fraction = seconds / (days_in_year * 24 * 3600)
                return np.cos(2 * np.pi * year_fraction)

             
            # Other derived features
            case F.POWER_RATIO:
                return api.get(F.PVLIB_DC_POWER) / api.get_const(F.DCP0)
            case F.COS_AOI:
                return np.cos(api.get(F.AOI))
            case F.TIME_SINCE_SUNLIGHT:
                df = api.get([F.TIME, F.DAY, F.PVLIB_POA_IRRADIANCE])
                sunrise = (
                    df[df[F.PVLIB_POA_IRRADIANCE.name] > 5]
                    .groupby(df[F.DAY.name])[F.TIME.name]
                    .first()
                )
                sunrise = sunrise.reindex(df[F.DAY.name].unique(), fill_value=df[F.TIME.name].min())
                day_to_sunrise = df[F.DAY.name].map(sunrise)
                result = (df[F.TIME.name] - day_to_sunrise).dt.total_seconds()/3600
                return result
            case F.RELATIVE_WIND_DIRECTION:
                return api.get(F.WIND_DIRECTION) - api.get_const(F.AZIMUTH)
            case F.WIND_NORMAL_COMPONENT:
                return api.get(F.WIND_SPEED) * np.cos(np.deg2rad(api.get(F.RELATIVE_WIND_DIRECTION))) * np.sin(np.deg2rad(api.get_const(F.TILT)))

            #new composed features
            case F.POA_COS_AOI:
                return api.get(F.PVLIB_POA_IRRADIANCE) * api.get(F.COS_AOI)
            case F.POA_WIND_SPEED:
                return api.get(F.PVLIB_POA_IRRADIANCE) * api.get(F.WIND_SPEED)
            case F.DCP_PER_AREA:
                return api.get(F.PVLIB_DC_POWER) / api.get(F.AREA)
            case F.DHI_PER_GHI:
                return api.get(F.DHI) / api.get(F.GHI)
            case F.GAMMA_TEMP_DIFFERENCE:
                return api.get(F.GAMMA) * (api.get(F.AIR_TEMP) - api.get(F.FAIMAN_MODULE_TEMP))
            case F.GAMMA_POA:
                return api.get(F.GAMMA) * api.get(F.PVLIB_POA_IRRADIANCE)
            case F.RELATIVE_AZIMUTH:
                return api.get(F.AZIMUTH) - api.get(F.SOLAR_AZIMUTH)
            


            case _:
                raise NotImplementedError

    # More complicated calculations of features via regressions
    @classmethod
    def calculate_faiman_coefficients(cls, api: FeatureAccessor, alpha: float = 0.9, poa_min: float = 50, poa_max: float = 1200) -> tuple[float, float]:
        """Calculate module temperature by the fainman model. Use default coefficients if not measured module temperature data is available."""
        if not api.available(F.PVDAQ_MODULE_TEMP):
            return 25.0, 6.84 # default values for u0, u1 from pvlib
        df_fit = api.copy([F.WIND_SPEED, F.PVLIB_POA_IRRADIANCE, F.TEMP_DIFFERENCE_MEASURED])
        df_fit.ftr.filter({
            F.PVLIB_POA_IRRADIANCE: (poa_min, poa_max),
            F.TEMP_DIFFERENCE_MEASURED: (3, 60)
        })
        return cls.faiman_regression(df_fit.ftr, alpha)
    
    @classmethod
    def faiman_regression(cls, api: FeatureAccessor, alpha: float = 0.9) -> tuple[float, float]:
        """Use linear regression to estimate the parameters for the fainman module temperature model
        module_temperature = air_temperature + (alpha * poa_irridiance) / (u0 + u1 * wind_speed)"""
        X_fit = api.get([F.WIND_SPEED])
        y_fit = alpha * api.get(F.PVLIB_POA_IRRADIANCE) / api.get(F.TEMP_DIFFERENCE_MEASURED)
        reg = LinearRegression().fit(X_fit, y_fit)
        return reg.intercept_, reg.coef_[0]  

    @classmethod
    def calculate_annual_dcp0_gamma(cls, api: FeatureAccessor,
                                    max_dcp_per_area: float = 150,
                                    poa_min: float = 200,
                                    poa_max: float = 1200) -> pd.DataFrame:
        area = api.get_const(F.AREA)
        physics_dcp_limit = area * max_dcp_per_area if area is not None else float('inf')
        dcp_filter_limit = min(api.get(F.PVDAQ_DC_POWER).quantile(0.99), physics_dcp_limit)
        df_fit = api.copy([F.YEAR, F.FAIMAN_MODULE_TEMP, F.PVLIB_POA_IRRADIANCE, F.PVDAQ_DC_POWER])
        df_fit = df_fit.ftr.filter({
            F.PVLIB_POA_IRRADIANCE: (poa_min, poa_max),
            F.PVDAQ_DC_POWER: (1, dcp_filter_limit)
        })
         # Necessary for averaging year dependend weather
        years = api.get(F.YEAR).unique()
        results = []
        
        for year in years:
            year_df = df_fit[df_fit[F.YEAR.name] == year]
            if len(year_df) < 30:
                continue # insufficient data for this year
            dcp0, gamma = cls.dcp0_gamma_regression(year_df.ftr)

            # limit to reasonable values (maximally about 150 W/m² per m² of PV area)
            if dcp0 <= 0 or dcp0 > physics_dcp_limit or gamma is None:
                 dcp0, gamma = None, None

            results.append({
                F.YEAR.name: year, 
                "annual_dcp0": dcp0, 
                "annual_gamma": gamma
            })
        if not results:
            return np.nan, np.nan
        results_df = pd.DataFrame(results).set_index(F.YEAR.name).sort_index()
        results_df.ftr.to_csv(Path("parameter") / "dcp0_gamma_per_year" / f"{api.get_const(F.SYSTEM_ID)}.csv")
        dcp0 = results_df['annual_dcp0'].mean()
        gamma = results_df['annual_gamma'].mean()
        return dcp0, gamma

    @classmethod
    def dcp0_gamma_regression(cls, api: FeatureAccessor) -> tuple[float | None, float | None]:
        # Input features for Linear Regression
        # dcp = pdc0 * (1 + gamma * (T_cell - 25)) * POA/1000
        X = (api.get(F.FAIMAN_MODULE_TEMP) - 25).values.reshape(-1,1)  # delta T
        poa = (api.get(F.PVLIB_POA_IRRADIANCE) / 1000).values.reshape(-1,1)
        X_fit = np.hstack([poa, poa*X])
        y_fit = api.get(F.PVDAQ_DC_POWER).values
        model = LinearRegression(fit_intercept=False)
        try:
            model.fit(X_fit, y_fit)
        except ValueError:
            return (None, None)

        dcp0 = model.coef_[0]
        gamma = model.coef_[1] / dcp0 if dcp0 != 0 else None

        return dcp0, gamma