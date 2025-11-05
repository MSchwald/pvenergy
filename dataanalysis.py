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

class PV_CONST:
    """Considered parameters to calculate from pvdaq solar data"""
    # module specific parameters to model the system's power output for given temperature and poa
    # dcp = pdc0 * (1 + gamma * (module_temperature - 25)) * poa_irridiance / 1000
    DCP0 = "dcp0" # The module's power under standard conditions (25°C air temperature, 1000 W/m² poa)
    GAMMA = "gamma" # temperature coefficient

    # Parameters of the Faiman model to calculate the module's temperature
    # module_temperature = air_temperature + (alpha * poa_irridiance) / (u0 + u1 * wind_speed)
    FAIMAN_U0 = "faiman_u0"
    FAIMAN_U1 = "faiman_u1"

class FEATURE_COL:
    """Column names for the derived features of pvdaq pv systems"""
    MODULE_TEMP_CALCULATED = "module_temp_calculated"
    TEMP_DIFFERENCE = "temp_difference" # difference between module and air temperature
    DAYS_SINCE_START = "days_since_start"
    DAY_OF_YEAR = "day_of_year"

class DataAnalysis:

    SYSTEM_CONST = pd.DataFrame(columns=[PV_CONST.DCP0, PV_CONST.GAMMA, PV_CONST.FAIMAN_U0, PV_CONST.FAIMAN_U1], index = Pvdaq.load_metadata().index)

    @classmethod
    def get_raw_training_data(cls, system_id: int):
        pv_data = Pvdaq.load_dcp_data(system_id)
        if pv_data.empty:
            print(f"No measured dcp data available for system ID {system_id}.")
            return
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
    def calculate_dcp0_gamma(cls, system_id: int, max_dcp_per_area: float = 150, poa_min: float = 200, poa_max: float = 1200) -> (float | None, float | None):
        if not pd.isna(cls.SYSTEM_CONST.loc[system_id, PV_CONST.DCP0]) and not pd.isna(cls.SYSTEM_CONST.loc[system_id, PV_CONST.GAMMA]):
            return cls.SYSTEM_CONST.loc[system_id, PV_CONST.DCP0], cls.SYSTEM_CONST.loc[system_id, PV_CONST.GAMMA]
        local_dir = Path("parameter/dcp0_gamma_per_year")
        file = local_dir / f"{system_id}.csv"
        if file.exists():
            df = pd.read_csv(file, index_col='year')
            DCP0_GLOBAL = df['annual_dcp0'].mean()
            GAMMA_GLOBAL = df['annual_gamma'].mean()
            return DCP0_GLOBAL, GAMMA_GLOBAL
        local_dir.mkdir(parents=True, exist_ok=True)
        
        area = Pvdaq.meta(system_id)[META_COL.AREA]
        df = cls.get_raw_training_data(system_id)
        if df is None:
            return

        df_fit = df[[PV_COL.TIME, PV_COL.DC_POWER, PV_COL.MODULE_TEMP, WEATHER_COL.POA_IRRADIANCE]]
        DCP_MAX_SYSTEM = area * max_dcp_per_area
        dcp_upper_limit = min(df_fit[PV_COL.DC_POWER].quantile(0.99), DCP_MAX_SYSTEM)
        df_fit = cls.filter_data(df_fit, {
            WEATHER_COL.POA_IRRADIANCE: (poa_min, poa_max),
            PV_COL.DC_POWER: (1,  dcp_upper_limit)
        })
        
        df_fit['year'] = df_fit[PV_COL.TIME].dt.year # Benötigt für die spätere Glättung
        years = df_fit['year'].unique()

        results = []

        for year in years:
            year_df = df_fit[df_fit['year'] == year].copy()
            if len(year_df) < 30:
                continue # insufficient data for this year
            dcp0, gamma = cls.dcp0_gamma_regression(year_df)

            # limit to reasonable values (maximally about 150 W/m² per m² of PV area)
            if dcp0 <= 0 or dcp0 > area * 150 or gamma is None:
                 dcp0, gamma = None, None

            results.append({
                "year": year, 
                "annual_dcp0": dcp0, 
                "annual_gamma": gamma
            })

        results_df = pd.DataFrame(results).set_index('year').sort_index()
        results_df.to_csv(file, index=True)

        DCP0_GLOBAL = results_df['annual_dcp0'].mean()
        GAMMA_GLOBAL = results_df['annual_gamma'].mean()
        cls.SYSTEM_CONST.loc[system_id, PV_CONST.DCP0] = DCP0_GLOBAL
        cls.SYSTEM_CONST.loc[system_id, PV_CONST.GAMMA] = GAMMA_GLOBAL
        return DCP0_GLOBAL, GAMMA_GLOBAL

    @classmethod
    def faiman_regression(cls, df: pd.DataFrame, alpha: float = 0.9, poa_min: float = 50, poa_max: float = 1200) -> tuple[float, float]:
        df_fit = df.copy()
        if not FEATURE_COL.TEMP_DIFFERENCE in df_fit.columns:
            df_fit[FEATURE_COL.TEMP_DIFFERENCE] = df_fit[FEATURE_COL.MODULE_TEMP_CALCULATED] - df_fit[WEATHER_COL.AIR_TEMP]
        y_fit = (alpha * df_fit[WEATHER_COL.POA_IRRADIANCE]) / df_fit[FEATURE_COL.TEMP_DIFFERENCE]
        X_fit = df_fit[[WEATHER_COL.WIND_SPEED]]
        
        reg = LinearRegression().fit(X_fit, y_fit)
        u0_fitted = reg.intercept_
        ua_fitted = reg.coef_[0]

        return u0_fitted, ua_fitted

    @classmethod
    def module_temperature_faiman(cls, df: pd.DataFrame, alpha: float = 0.9, poa_min: float = 50, poa_max: float = 1200) -> pd.DataFrame:
        df_fit = df.copy()
        df_fit[FEATURE_COL.TEMP_DIFFERENCE] = df_fit[FEATURE_COL.MODULE_TEMP_CALCULATED] - df_fit[WEATHER_COL.AIR_TEMP]
        df_fit = cls.filter_data(df_fit, {
            WEATHER_COL.POA_IRRADIANCE: (poa_min, poa_max),
            FEATURE_COL.TEMP_DIFFERENCE: (3, 60)
        })

        y_fit = (alpha * df_fit[WEATHER_COL.POA_IRRADIANCE]) / df_fit[FEATURE_COL.TEMP_DIFFERENCE]
        X_fit = df_fit[[WEATHER_COL.WIND_SPEED]]
        
        reg = LinearRegression().fit(X_fit, y_fit)
        u0_fitted = reg.intercept_
        ua_fitted = reg.coef_[0]

        print("\n--- Faiman Koeffizienten Regression ---")
        print(f"Gefitteter u0 (Koeffizient ohne Wind): {u0_fitted:.4f}")
        print(f"Gefitteter ua (Wind-abhängiger Koeffizient): {ua_fitted:.4f}")

        # module_temperature = air_temperature + (alpha * poa_irridiance) / (u0 + u1 * wind_speed)
        return pd.Series(
            df[WEATHER_COL.AIR_TEMP] + 
            (alpha * df[WEATHER_COL.POA_IRRADIANCE]) / 
            (u0_fitted + ua_fitted * df[WEATHER_COL.WIND_SPEED])
        )

    @classmethod
    def train_random_forest_model(cls, system_id: int, features: list[str] | None):
        df = cls.get_raw_training_data(system_id)
        if df.empty:
            print(f"No measured dcp data available for system ID {system_id}.")
            return

        # Calculation of additional features if not already present
        if FEATURE_COL.MODULE_TEMP_CALCULATED in features and FEATURE_COL.MODULE_TEMP_CALCULATED not in df.columns:
            df[FEATURE_COL.MODULE_TEMP_CALCULATED] = cls.module_temperature_faiman(df)
        
        df = df[PV_COL.DC_POWER].clip(lower=0)
        df = cls.filter_data(df, {PV_COL.DC_POWER: (0,3000)})

        if FEATURE_COL.DAYS_SINCE_START in features and FEATURE_COL.DAYS_SINCE_START not in df.columns:
            starting_day = df[PV_COL.TIME].head(1)
            df[FEATURE_COL.DAYS_SINCE_START] = (df[PV_COL.TIME] - starting_day).dt.days

        if FEATURE_COL.DAY_OF_YEAR in features and FEATURE_COL.DAY_OF_YEAR not in df.columns:
            df[FEATURE_COL.DAY_OF_YEAR] = df[PV_COL.TIME].dt.dayofyear

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