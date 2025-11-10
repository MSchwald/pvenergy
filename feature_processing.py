from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Any
import pandas as pd
import numpy as np
from enum import Enum
from feature_catalog import Feature, Source
from feature_catalog import FeatureCatalog as F
from sklearn.linear_model import LinearRegression
import pvlib
import pytz
from timezonefinder import TimezoneFinder

class FeatureProcessing:
    ALL_FEATURES: list[Feature] = [
        feature for feature in vars(F).values() if isinstance(feature, Feature)
    ]
    ALL_FEATURE_NAMES: list[str] = [feature.name for feature in ALL_FEATURES]
    CALCULATED_FEATURES: list[Feature] = [
        feature for feature in ALL_FEATURES if feature.source == Source.CALCULATED
    ]

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
            case F.SOLAR_ZENITH | F.SOLAR_AZIMUTH:
                solpos = pvlib.solarposition.get_solarposition(time = api.get(F.LOCALIZED_TIME),
                                                        latitude = api.get_const(F.LATITUDE),
                                                        longitude = api.get_const(F.LONGITUDE),
                                                        altitude = api.get_const(F.ELEVATION))
                api.set(F.SOLAR_AZIMUTH, solpos["azimuth"])
                api.set(F.SOLAR_ZENITH, solpos["apparent_zenith"])
                return api.get(feature)
            case F.PVLIB_POA_IRRADIANCE:
                poa = pvlib.irradiance.get_total_irradiance(surface_tilt = api.get_const(F.TILT),
                                                surface_azimuth = api.get_const(F.AZIMUTH),
                                                solar_zenith = api.get(F.SOLAR_ZENITH),
                                                solar_azimuth = api.get(F.SOLAR_AZIMUTH),
                                                dni = api.get(F.DNI),
                                                ghi = api.get(F.GHI),
                                                dhi = api.get(F.DHI),
                                                albedo = api.get(F.SURFACE_ALBEDO))
                return poa["poa_global"]
            case F.AOI:
                return pvlib.irradiance.aoi(surface_tilt = api.get_const(F.TILT),
                                surface_azimuth = api.get_const(F.AZIMUTH),
                                solar_zenith = api.get(F.SOLAR_ZENITH),
                                solar_azimuth = api.get(F.SOLAR_AZIMUTH))
    
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
            case F.HOUR:
                return api.get(F.TIME).dt.hour
            # Time features to model degradation, seasonal soiling and daily heat inertia
            case F.DAYS_SINCE_START:
                starting_day = api.get(F.TIME).iloc[0]
                return (api.get(F.TIME) - starting_day).dt.days
            case F.DAY_OF_YEAR:
                return api.get(F.TIME).dt.dayofyear
            # Other derived features
            case F.POWER_RATIO:
                return api.get(F.PVDAQ_DC_POWER) / api.get_const(F.DCP0)
            case F.COS_AOI:
                return np.cos(api.get(F.AOI))
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
    def calculate_annual_dcp0_gamma(cls, api: FeatureAccessor, max_dcp_per_area: float = 150, poa_min: float = 200, poa_max: float = 1200) -> pd.DataFrame:
        area = api.get_const(F.AREA)
        physics_dcp_limit = area * max_dcp_per_area if area is not None else float('inf')
        dcp_filter_limit = min(api.get(F.PVDAQ_DC_POWER).quantile(0.99), physics_dcp_limit)
        df_fit = api.copy([F.YEAR, F.FAIMAN_MODULE_TEMP, F.PVLIB_POA_IRRADIANCE, F.PVDAQ_DC_POWER])
        df_fit = df_fit.ftr.filter({
            F.PVLIB_POA_IRRADIANCE: (poa_min, poa_max),
            F.PVDAQ_DC_POWER: (1, dcp_filter_limit)
        })
        print(df_fit.info)
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
        y_fit = api.get(F.PVDAQ_DC_POWER).values.reshape(-1,1)
        model = LinearRegression(fit_intercept=False)
        try:
            model.fit(X_fit, y_fit)
        except ValueError:
            return (None, None)

        dcp0 = model.coef_[0]
        gamma = model.coef_[1] / dcp0 if dcp0 != 0 else None

        return dcp0, gamma