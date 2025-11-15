import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import pvlib
import data_request
from data_request import Pvdaq
from feature_catalog import FeatureCatalog as F
from feature_catalog import Feature, Source
from typing import Union, Callable
from tqdm import tqdm
from dataclasses import dataclass
import joblib

FeatureList = Union[Feature, tuple[Feature], list[Feature], None] 

class Scaler:
    STANDARD = sklearn.preprocessing.StandardScaler
    MINMAX = sklearn.preprocessing.MinMaxScaler
    ROBUST = sklearn.preprocessing.RobustScaler
    MAXABS = sklearn.preprocessing.MaxAbsScaler

class EvaluationMethod:
    """Methods to analyze the performance of trained ML models"""
    @staticmethod
    def rmse(model, X_test, y_test, y_pred):
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return "Root Mean Squared Error (RMSE)", rmse
    
    @staticmethod
    def r2(model, X_test, y_test, y_pred):
        score = r2_score(y_test, y_pred)
        return "R² Score", score
    
    @staticmethod
    def feature_importance(model, X_test, y_test, y_pred):
        df = pd.DataFrame({
            'Feature': X_test.columns.tolist(),
            'Importance': model.feature_importances_
        }).sort_values(by = 'Importance', ascending = False)
        return "Feature Importance Analysis:\n", df

@dataclass
class Model:
    """Defining properties of ML models"""
    name: str
    estimator: Callable
    scaler: Scaler | None = None
    evaluation_methods: list[EvaluationMethod] | None = None
    _trained_model: object | None = None
    _fitted_scaler: object | None = None

    def __str__(self):
        return self.name

    def apply_scaler(self, X: pd.DataFrame, train: bool = False) -> pd.DataFrame:
        """
        Rescales given dataframe with the models individual scaler.
        If train == True, adapt the scaler to the given data for preparing the model
        to make predictions on future data using the same scaling properties.
        """
        if self.scaler is None:
            return X
        if train:
            self._fitted_scaler = self.scaler()
            data = self._fitted_scaler.fit_transform(X)
        else:
            if self._fitted_scaler is None:
                raise RuntimeError(f"Scaler of model {self.name} has not been trained yet.")
            data = self._fitted_scaler.transform(X)
        return pd.DataFrame(data = data, columns = X.columns, index = X.index)

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        # For models specific rescaling of the training data is important for their performance
        X_train_scaled = self.apply_scaler(X_train, train = True)
        model = self.estimator()
        model.fit(X_train_scaled, y_train)
        self._trained_model = model

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if self._trained_model is None:
            raise RuntimeError(f"Model {self.name} has not been trained yet.")
        X_test_scaled = self.apply_scaler(X_test)
        return self._trained_model.predict(X_test_scaled)

    def evaluate_performance(self, X_test, y_test, y_pred):
        results = []
        for method in self.evaluation_methods:
            results.append(method(self._trained_model, X_test, y_test, y_pred))
        return results

    def save_trained_model(self, path: Path):
        path.parent.mkdir(parents = True, exist_ok = True)
        joblib.dump({
            "model": self._trained_model,
            "scaler": self._fitted_scaler
        }, path)

    def load_trained_model(self, path: Path):
        data = joblib.load(path)
        self._trained_model = data["model"]
        self._fitted_scaler = data["scaler"]

class ML_MODELS:
    """Collection ML models suitable for analyzing PVDAQ data"""
    RANDOM_FOREST = Model(
        name = "random_forest",
        estimator = lambda: RandomForestRegressor(n_estimators=200,
                                                random_state=42,
                                                n_jobs=-1),
        scaler = None,
        evaluation_methods = [EvaluationMethod.rmse,
                              EvaluationMethod.r2,
                              EvaluationMethod.feature_importance]
    )

def pipeline(system_ids: list[int],
            training_features: list[Feature],
            target_feature: Feature = F.PVDAQ_DC_POWER,
            clip_features: dict[Feature, tuple[int | None, int | None]] = {},
            filter_features: dict[Feature, tuple[int | None, int | None]] = {},
            ml_model: Model = ML_MODELS.RANDOM_FOREST,
            file_limit: int | None = None, # Safety option and for quick testing pipeline with few data
            mute_tqdm = False, # Reduce info on data processing printed by the console
            log_raw_data: bool = False):
    """
    Train a given ML-model on pv and weather data for the PVDAQ pv systems
    of the given system IDs. Provide: Training and target features from
    the FeatureCatalog, clipping and filtering options (to be applied before training).
    Pipeline prints an analysis on the quality of the model and ranks the importance of
    the provided training features; returns a predictive model that can be used
    for further forecastings on new data.

    """
    # Prepare list of all requested features
    features: list[Feature] = training_features.copy()
    if target_feature not in features:
        features.append(target_feature)

    # Download pv and weather data and calculate requested features for the given pv systems
    dfs = []
    for system_id in tqdm(system_ids, desc=f"Loading PVDAQ and NSRDB data"):
        tqdm.write(f"Loading data for system {system_id}")
        df = data_request.get_features(system_id = system_id, features = features,
                                file_limit = file_limit, mute_tqdm = mute_tqdm)
        if log_raw_data:
            df.ftr.to_csv(Path("test.csv"))
        # Clean up data for each system individually
        df = df.ftr.clip(clip_features)
        df = df.ftr.filter(filter_features) # also drops rows with NaNs
        if not df.empty:
            dfs.append(df)
    df_full: pd.DataFrame = pd.concat(dfs, ignore_index = True)

    # Split data into training data and target feature
    X: pd.DataFrame = df_full.ftr.get([ftr for ftr in features if ftr != target_feature])
    y: pd.Series = df_full.ftr.get(target_feature)
    test_size = 0.2
    print(f"\nChoosing a random {int(100*(1-test_size))}:{int(100*test_size)} split of the data:")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_size,
        random_state = 42 # chooses a random seed, only for reproducibility of the results
        # (its value is supposed to have no significant influence on the model's quality.)
    )
    print(f"  {X_train.shape[0]} samples for training data.")
    print(f"  {X_test.shape[0]} samples to testing data.")

    print(f"\nTraining ML-model '{ml_model.name}' to predict feature '{target_feature}'.")                      
    ml_model.train_model(X_train, y_train)

    print(f"Testing trained model on predicting '{target_feature}' from the randomly chosen testing data.")
    y_pred = ml_model.predict(X_test)

    print("\nAnalysis of the model's performance on the testing data:")
    results = ml_model.evaluate_performance(X_test, y_test, y_pred)
    for name, result in results:
        print(name, result)

    saving_path = Path("trained_models")
    print(f"\nSaving the model's trained state to {saving_path}.")
    ml_model.save_trained_model(Path("trained_models") / f"{ml_model.name}.joblib")

if __name__ == "__main__":
    """Testing space for the pipeline"""
    features: list[Feature] = [F.POWER_RATIO, F.PVLIB_POA_IRRADIANCE, F.DAY_OF_YEAR, F.TIME_SINCE_SUNLIGHT, F.NSRDB_CLEAR_SKY_RATIO, F.COS_AOI]
    # Chooses all system ids where all important features are recorded
    # to predict the default target feature PVDAQ_DC_POWER
    system_ids = Pvdaq.filter_systems(metacols = [F.PVDAQ_DC_POWER, F.PVDAQ_MODULE_TEMP])
    ml_model = ML_MODELS.RANDOM_FOREST
    df = pipeline(system_ids = [2],
                    training_features = features,
                    target_feature = F.PVDAQ_DC_POWER,
                    clip_features = {F.PVDAQ_DC_POWER: (0, None)},
                    filter_features = {F.PVDAQ_DC_POWER: (0, 3000)},
                    ml_model = ml_model,
                    file_limit = None, mute_tqdm = False,
                    log_raw_data = True)