import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from weatherdata import Nsrdb, WEATHER_COL
from solardata import Pvdaq, META_COL, PV_COL
from pathlib import Path
import pvlib

class PV_CONST:
    """Considered parameters to calculate from pvdaq solar data"""
    # module specific parameters to model the system's power output for given temperature and poa
    # dcp = pdc0 * (1 + gamma * (module_temperature - 25)) * poa_irridiance / 1000
    DCP0 = "dcp0" # The module's power under standard conditions (25°C air temperature, 1000 W/m² poa)
    GAMMA = "gamma" # temperature coefficient

    # Parameters of the Faiman model to calculate the module's temperature
    # module_temperature = air_temperature + (alpha * poa_irridiance) / (u0 + u1 * wind_speed)
    FAIMAN_U0 = "faiman_u0" # General Heat Loss Coefficient ("Convective Heat Loss Coefficient at Zero Wind Speed")
    FAIMAN_U1 = "faiman_u1" # Wind-Dependent Heat Loss Coefficient ("Wind Speed Coefficient")

class FEATURE_COL:
    """Column names for the derived features of pvdaq pv systems"""
    MODULE_TEMP_CALCULATED = "module_temp_calculated"
    TEMP_DIFFERENCE_MEASURED = "temp_difference_measured" # difference between measured module and air temperature
    DAYS_SINCE_START = "days_since_start"
    DAY_OF_YEAR = "day_of_year"
    POWER_RATIO = "power_ratio"
    GAMMA = "gamma"

class DataAnalysis:
    """Calculate featurs for pvdaq pv system and analyze data with machine learning methods"""
    SYSTEM_CONST = pd.DataFrame(data = None,
                                columns = [PV_CONST.DCP0, PV_CONST.GAMMA,
                                PV_CONST.FAIMAN_U0, PV_CONST.FAIMAN_U1],
                                index = Pvdaq.load_metadata().index)
    
    @classmethod
    def get_raw_training_data(cls, system_id: int):
        pv_data = Pvdaq.load_dcp_module_temp_data(system_id)
        if pv_data.empty:
            print(f"No measured dcp data available for system ID {system_id}.")
            return pd.DataFrame()
        start, end = pv_data[PV_COL.TIME].iloc[1], pv_data[PV_COL.TIME].iloc[-1]
        weather_data = Nsrdb.load_system(system_id, start, end)
        df = pd.merge(pv_data, weather_data, left_on=PV_COL.TIME, right_on=WEATHER_COL.TIME, how='inner').drop(columns=[WEATHER_COL.TIME])
        df.to_csv("test.csv", index=False)
        return df
    
    @classmethod
    def filter_data(cls, data: pd.DataFrame, min_max_dict: dict) -> pd.DataFrame:
        for column, (min_value, max_value) in min_max_dict.items():
            data = data[(data[column] >= min_value) & (data[column] <= max_value)]
        return data.dropna(how="any")

    @classmethod
    def dcp0_gamma_regression(cls, df: pd.DataFrame) -> tuple[float | None, float | None]:
        # Input features for Linear Regression
        # dcp = pdc0 * (1 + gamma * (T_cell - 25)) * POA/1000
        X = (df[FEATURE_COL.MODULE_TEMP_CALCULATED] - 25).values.reshape(-1,1)  # delta T
        poa = (df[WEATHER_COL.POA_IRRADIANCE] / 1000).values.reshape(-1,1)
        X_fit = np.hstack([poa, poa*X])
        y_fit = df[PV_COL.DC_POWER].values

        model = LinearRegression(fit_intercept=False)
        try:
            model.fit(X_fit, y_fit)
        except ValueError:
            return (None, None)

        dcp0 = model.coef_[0]
        gamma = model.coef_[1] / dcp0 if dcp0 != 0 else None

        return (dcp0, gamma)

    @classmethod
    def calculate_annual_dcp0_gamma(cls, df: pd.DataFrame, area = None, max_dcp_per_area: float = 150, poa_min: float = 200, poa_max: float = 1200) -> pd.DataFrame:
        physics_dcp_limit = area * max_dcp_per_area if area is not None else float('inf')
        dcp_filter_limit = min(df[PV_COL.DC_POWER].quantile(0.99), physics_dcp_limit) 
        df_fit = cls.filter_data(df, {
            WEATHER_COL.POA_IRRADIANCE: (poa_min, poa_max),
            PV_COL.DC_POWER: (1, dcp_filter_limit)
        }).copy()
        df_fit['year'] = df_fit[PV_COL.TIME].dt.year # Necessary for averaging year dependend weather
        years = df_fit['year'].unique()
        results = []
        
        for year in years:
            year_df = df_fit[df_fit['year'] == year].copy()
            if len(year_df) < 30:
                continue # insufficient data for this year
            dcp0, gamma = cls.dcp0_gamma_regression(year_df)

            # limit to reasonable values (maximally about 150 W/m² per m² of PV area)
            if dcp0 <= 0 or dcp0 > physics_dcp_limit or gamma is None:
                 dcp0, gamma = None, None

            results.append({
                "year": year, 
                "annual_dcp0": dcp0, 
                "annual_gamma": gamma
            })
        results_df = pd.DataFrame(results).set_index('year').sort_index()
        return results_df

    @classmethod
    def faiman_regression(cls, df: pd.DataFrame, alpha: float = 0.9) -> tuple[float, float]:
        """Use linear regression to estimate the parameters for the fainman module temperature model"""
        # module_temperature = air_temperature + (alpha * poa_irridiance) / (u0 + u1 * wind_speed)
        y_fit = (alpha * df[WEATHER_COL.POA_IRRADIANCE]) / df[FEATURE_COL.TEMP_DIFFERENCE_MEASURED]
        X_fit = df[[WEATHER_COL.WIND_SPEED]]
        reg = LinearRegression().fit(X_fit, y_fit)
        return reg.intercept_, reg.coef_[0]  

    @classmethod
    def calculate_faiman_coefficients(cls, df: pd.DataFrame, alpha: float = 0.9, poa_min: float = 50, poa_max: float = 1200) -> tuple[float, float]:
        """Calculate module temperature by the fainman model. Use default coefficients if not measured module temperature data is available."""
        if not PV_COL.MODULE_TEMP in df.columns:
            return 25.0, 6.84 # default values for u0, u1 from pvlib
        df_fit = df.copy()
        df_fit[FEATURE_COL.TEMP_DIFFERENCE_MEASURED] = df_fit[PV_COL.MODULE_TEMP] - df_fit[WEATHER_COL.AIR_TEMP]
        df_fit = cls.filter_data(df_fit, {
            WEATHER_COL.POA_IRRADIANCE: (poa_min, poa_max),
            FEATURE_COL.TEMP_DIFFERENCE_MEASURED: (3, 60)
        })
        return cls.faiman_regression(df_fit, alpha)

    @classmethod
    def calculate_pv_constants(cls, system_id: int):
        """Calculates and saves the constants dcp0, gamma, u0, u1 for a given pvdaq system"""
        meta = Pvdaq.meta(system_id)
        dcp0 = cls.SYSTEM_CONST.loc[system_id, PV_CONST.DCP0]
        gamma = cls.SYSTEM_CONST.loc[system_id, PV_CONST.GAMMA]
        u0 = cls.SYSTEM_CONST.loc[system_id, PV_CONST.FAIMAN_U0]
        u1 = cls.SYSTEM_CONST.loc[system_id, PV_CONST.FAIMAN_U1]
        if any(c is None for c in (dcp0, gamma, u0, u1)):
            df = cls.get_raw_training_data(system_id)
            if df.empty:
                cls.SYSTEM_CONST.loc[system_id, PV_CONST.DCP0] = pd.nan
                cls.SYSTEM_CONST.loc[system_id, PV_CONST.GAMMA] = pd.nan
                cls.SYSTEM_CONST.loc[system_id, PV_CONST.FAIMAN_U0] = pd.nan
                cls.SYSTEM_CONST.loc[system_id, PV_CONST.FAIMAN_U1] = pd.nan
                return None, None, None, None
        if dcp0 is None or gamma is None:
            local_dir = Path("parameter/dcp0_gamma_per_year")
            file = local_dir / f"{system_id}.csv"
            if file.exists():
                df_result = pd.read_csv(file, index_col='year')
            else:
                local_dir.mkdir(parents=True, exist_ok=True)
                df_fit = df[[PV_COL.TIME, PV_COL.DC_POWER, PV_COL.MODULE_TEMP, WEATHER_COL.POA_IRRADIANCE]].copy()
                df_result = cls.calculate_annual_dcp0_gamma(df_fit, meta[META_COL.AREA]) 
                df_result.to_csv(file, index=True)
            dcp0 = df_result['annual_dcp0'].mean()
            gamma = df_result['annual_gamma'].mean()
            cls.SYSTEM_CONST.loc[system_id, PV_CONST.DCP0] = dcp0 if dcp0 is not None else pd.nan
            cls.SYSTEM_CONST.loc[system_id, PV_CONST.GAMMA] = gamma if gamma is not None else pd.nan
        if u0 is None or u1 is None:
            u0, u1 = cls.calculate_faiman_coefficients(df)
            cls.SYSTEM_CONST.loc[system_id, PV_CONST.FAIMAN_U0] = u0 if u0 is not None else pd.nan
            cls.SYSTEM_CONST.loc[system_id, PV_CONST.FAIMAN_U1] = u1 if u1 is not None else pd.nan
        return dcp0, gamma, u0, u1

    @classmethod
    def module_temperature_faiman(cls, df: pd.DataFrame, u0 = 25.0, u1 = 6.84):
        return pvlib.temperature.faiman(df[WEATHER_COL.POA_IRRADIANCE], df[WEATHER_COL.AIR_TEMP], df[WEATHER_COL.WIND_SPEED], u0, u1)

    @classmethod
    def train_random_forest_model(cls, system_id: int, features: list[str] | None):
        df = cls.get_raw_training_data(system_id)
        if df.empty:
            print(f"No measured dcp data available for system ID {system_id}.")
            return
        df = df[PV_COL.DC_POWER].clip(lower=0)
        df = cls.filter_data(df, {PV_COL.DC_POWER: (0,3000)})

        dcp0, gamma, u0, u1 = cls.calculate_pv_constants(system_id)

        # Calculation of additional features if not already present
        for feature in [FEATURE_COL.MODULE_TEMP_CALCULATED, FEATURE_COL.DAYS_SINCE_START, FEATURE_COL.DAY_OF_YEAR]:
            if feature in features and feature not in df.columns:
                match feature:
                    case FEATURE_COL.MODULE_TEMP_CALCULATED:
                        df[feature] = cls.module_temperature_faiman(df, u0, u1)
                    case FEATURE_COL.DAYS_SINCE_START:
                        starting_day = df[PV_COL.TIME].head(1)
                        df[FEATURE_COL.DAYS_SINCE_START] = (df[PV_COL.TIME] - starting_day).dt.days
                    case FEATURE_COL.DAY_OF_YEAR:
                        df[FEATURE_COL.DAY_OF_YEAR] = df[PV_COL.TIME].dt.dayofyear
                    case FEATURE_COL.POWER_RATIO:
                        df[FEATURE_COL.POWER_RATIO] = df[PV_COL.DC_POWER] / dcp0
                    case FEATURE_COL.GAMMA:
                        df[FEATURE_COL.GAMMA] = gamma

        y = df[PV_COL.DC_POWER]
        X = df[features]

        # Train test split of the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2,
            random_state=42  # for reproducible results
        )
        print(f"Trainingsdatensatzgröße (X_train): {X_train.shape[0]} Messungen")
        print(f"Testdatensatzgröße (X_test): {X_test.shape[0]} Messungen")

        # Model training
        model = RandomForestRegressor(n_estimators=100, # number of trees in the random forest
                                      random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # Model evaluation
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print("-" * 30)
        print("Model evaluation on testing data:")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"  R² Score: {r2:.4f}")
        print("-" * 30)

        # Feature Importance Analysis
        importances = model.feature_importances_
        feature_names = X.columns # Liste der Feature-Namen
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        print(feature_importance_df)
