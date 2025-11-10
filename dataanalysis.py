import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import pvlib
from data_request import request_data
from feature_catalog import FeatureCatalog as F
from feature_catalog import Feature
from typing import Union

FeatureList = Union[Feature, list[Feature], None] 

class DataAnalysis:

    @classmethod
    def train_random_forest_model(cls, system_id: int,
                                  features: FeatureList = None,
                                  file_limit: int | None = None,
                                  mute_tqdm = False):
        """
        Train a RFM on estimating the power output (PVDAQ_DC_POWER)
        of given PVDAQ system by a given list of features.
        Analyze the model's performance and rank the importance of the features.
        """
        if F.PVDAQ_DC_POWER not in features:
            features.append(F.PVDAQ_DC_POWER)
        df = request_data(system_id, file_limit = file_limit, mute_tqdm = mute_tqdm)        
        if df.empty:
            print(f"No measured data available for system ID {system_id}.")
            return
        df = df.ftr.get(features)

        df = df.ftr.clip({F.PVDAQ_DC_POWER: (0, None)})
        df = df.ftr.filter({F.PVDAQ_DC_POWER: (0, 3000)})

        y = df.ftr.get(F.PVDAQ_DC_POWER)
        X = df.ftr.get([ftr for ftr in features if ftr != F.PVDAQ_DC_POWER])

        # Randomize train test split of given ratios of sizes
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2,
            random_state=42  # random seed for reproducible results
        )
        print(f"Training data size: {X_train.shape[0]}")
        print(f"Test data size: {X_test.shape[0]}")

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
        feature_names = [ftr.name for ftr in features if ftr != F.PVDAQ_DC_POWER]
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        print(feature_importance_df)

if __name__ == "__main__":
    DataAnalysis.train_random_forest_model(system_id = 2, features = [F.PVLIB_POA_IRRADIANCE, F.COS_AOI, F.WIND_SPEED, F.AIR_TEMP, F.FAIMAN_MODULE_TEMP, F.HOUR])
