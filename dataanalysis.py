# data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# machine learning
import sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import lightgbm as lgb

# modules for working with features
import data_request
from data_request import request_data
from feature_catalog import FeatureCatalog as F
from feature_catalog import Feature
from feature_processing import FeatureProcessing as fp

# python basics
from pathlib import Path
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
    # search for best hyperparmeters with RandomizedSearchCV
    hyperparam_grid: dict | None = None # possible hyperparam combinations to choose from
    n_iter_search: int = 10 # amount of random combinations to compare
    # trained model gets saved here for further use
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

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, hyper_parameter_search: bool = True) -> None:
        # For some models specific rescaling of the training data is important for their performance
        X_train_scaled = self.apply_scaler(X_train, train = True)
        model = self.estimator()

        if hyper_parameter_search and self.hyperparam_grid is not None:
            print(f"Search for best hyperparameters for {self.name}...")
            search = RandomizedSearchCV(
                estimator = model,
                param_distributions = self.hyperparam_grid,
                n_iter = self.n_iter_search,
                scoring = 'neg_root_mean_squared_error',
                cv = 3, # number of cross validation folds
                n_jobs = -1, # use all CPU kernels
                verbose = 1, # show progress
                random_state = 42
            )
            search.fit(X_train_scaled, y_train)
            self._trained_model = search.best_estimator_
            print(f"Best hyperparameters: {search.best_params_}")
        else:
            # Standard training
            model.n_jobs = -1
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
                                                random_state=42),
        scaler = None,
        evaluation_methods = [EvaluationMethod.rmse,
                              EvaluationMethod.r2,
                              EvaluationMethod.feature_importance],
        hyperparam_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.5]
        },
        n_iter_search = 10
    )
    XGBOOST = Model(
        name = "xgboost",
        estimator = lambda: XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        scaler = None,
        evaluation_methods = [EvaluationMethod.rmse,
                              EvaluationMethod.r2,
                              EvaluationMethod.feature_importance],
        hyperparam_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 1, 2, 5],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [0.5, 1, 1.5, 2]
        },
        n_iter_search=10  
    )
    LIGHTGBM = Model(
        name = "lightgbm",
        estimator = lambda: lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            num_leaves = 50,
            random_state=42
        ),
        scaler = None,
        evaluation_methods = [EvaluationMethod.rmse,
                              EvaluationMethod.r2,
                              EvaluationMethod.feature_importance],
        hyperparam_grid={
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'num_leaves': [31, 50, 70],
        },
        n_iter_search=10
    )

def get_system_constants(system_ids: list[int], constant_features: list[Feature]) -> pd.DataFrame:
    """Calculate relevant system constants and cache the result to save time."""
    constant_file = Path("system_constants.csv")
    if constant_file.exists():
        return pd.read_csv(constant_file, index_col = 0)
    pd.DataFrame(data = np.nan, columns=[ftr.name for ftr in constant_features], index = system_ids)
    for id in tqdm(system_ids, desc="Calculating system constants..."):
        tqdm.write(f"for system {id}")
        df = request_data(id)
        for ftr in constant_features:
            system_constants.loc[id, ftr.name] = df.ftr.get_const(ftr)
        print(df.ftr.get_const(constant_features))
        system_constants.to_csv(constant_file, index = True)

def get_training_data(
    system_ids: list[int],
    clip_features: dict[Feature, tuple[int | None, int | None]] = {},
    filter_features: dict[Feature, tuple[int | None, int | None]] = {},
    file_limit: int | None = None, # Safety option and for quick testing pipeline with few data
    mute_tqdm = False, # Reduce info on data processing printed by the console
    training_data_cache: str | None = None, # Cache training data before machine learning
    use_cached_training_data: bool = True) -> pd.DataFrame:

    if training_data_cache is not None and Path(training_data_cache).exists() and use_cached_training_data:
        df_full = pd.read_parquet(training_data_cache)
        #df_full.set_index(F.TIME.name)
    else:
        # Use some already calculated constant features to save time
        system_constants = cls.get_system_constants(
            system_ids = system_ids,
            constant_features = [F.DCP0, F.GAMMA, F.FAIMAN_U0, F.FAIMAN_U1])
        ).dropna(how = "any")
        cached_ids = system_constants.index.to_list()
        cached_constant_features = [fp.FEATURE_FROM_NAME[col] for col in system_constants.columns]
        # Download pv and weather data and calculate requested features for the given pv systems
        dfs = []
        for system_id in tqdm(system_ids, desc=f"Loading PVDAQ and NSRDB data"):
            tqdm.write(f"Loading data for system {system_id}")
            df = data_request.request_data(system_id = system_id, file_limit = file_limit, mute_tqdm = mute_tqdm)
            if system_id in cached_ids:
                for ftr in cached_constant_features:
                    df.ftr.set_const({ftr: system_constants.loc[system_id, ftr.name]})
            df = df.ftr.get(features)
            # Clean up data for each system individually
            df = df.ftr.clip(clip_features)
            df = df.ftr.filter(filter_features) # also drops rows with NaNs
            if not df.empty:
                dfs.append(df)
        df_full: pd.DataFrame = pd.concat(dfs)

        if training_data_cache is not None:
            df_full.to_parquet(training_data_cache)

def pipeline(system_ids: list[int],
            training_features: list[Feature],
            target_feature: Feature = F.PVDAQ_DC_POWER,
            clip_features: dict[Feature, tuple[int | None, int | None]] = {},
            filter_features: dict[Feature, tuple[int | None, int | None]] = {},
            ml_model: Model = ML_MODELS.RANDOM_FOREST,
            hyper_parameter_search: bool = True,
            file_limit: int | None = None, # Safety option and for quick testing pipeline with few data
            mute_tqdm = False, # Reduce info on data processing printed by the console
            training_data_cache: str | None = None, # Cache training data before machine learning
            use_cached_training_data: bool = True
):
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

    df_full = cls.get_training_data(
        system_ids = system_ids, clip_features = clip_features, filter_features = filter_features,
        file_limit = file_limit, mute_tqdm = mute_tqdm,
        training_data_cache = training_data_cache, use_cached_training_data = use_cached_training_data
    )

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
    ml_model.train_model(X_train, y_train, hyper_parameter_search)

    print(f"Testing trained model on predicting '{target_feature}' from the randomly chosen testing data.")
    y_pred = ml_model.predict(X_test)

    print("\nAnalysis of the model's performance on the testing data:")
    results = ml_model.evaluate_performance(X_test, y_test, y_pred)
    for name, result in results:
        print(name, result)
    
    saving_path = Path("trained_models")
    print(f"\nSaving the model's trained state to {saving_path}.")
    ml_model.save_trained_model(Path("trained_models") / f"{ml_model.name}_all_ids.joblib")

    return results

if __name__ == "__main__":
    """Testing space for the pipeline"""

    # Chooses all system ids where all important features are recorded to predict the default target feature PVDAQ_DC_POWER
    #ids = [id for id in Pvdaq.filter_systems(metacols = [F.PVDAQ_DC_POWER, F.PVDAQ_MODULE_TEMP, F.TILT, F.AZIMUTH])]
    
    # Current Training features for the ML models
    features: list[Feature] = [F.POWER_RATIO, F.PVLIB_POA_IRRADIANCE,
                               F.DAY_OF_YEAR, F.TIME_SINCE_SUNLIGHT,
                               F.NSRDB_CLEAR_SKY_RATIO, F.COS_AOI, F.WIND_NORMAL_COMPONENT,
                               F.POA_COS_AOI, F.POA_WIND_SPEED, F.DHI_PER_GHI,
                               F.DCP_PER_AREA, F.GAMMA_TEMP_DIFFERENCE, F.RELATIVE_AZIMUTH]
    
    # Choice of model to train
    ml_model = ML_MODELS.RANDOM_FOREST
    #RANDOM_FOREST, XGBOOST, LIGHTGBM
    
    # Save results?
    """
    results = pd.DataFrame(index = ids, columns = ["RMSE", "R2", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "F1", "F2", "F3", "F4", "F5", "F6", "F7"])
    for id in ids:
        res = pipeline(system_ids = [id],
                    training_features = features,
                    target_feature = F.PVDAQ_DC_POWER,
                    clip_features = {F.PVDAQ_DC_POWER: (0, None)},
                    filter_features = {F.PVDAQ_DC_POWER: (0, 3000)},
                    ml_model = ml_model,
                    file_limit = None, mute_tqdm = False,
                    log_raw_data = True)
        results.loc[id] = [res[0][1], res[1][1]] + res[2][1]["Importance"].to_list() + res[2][1]["Feature"].to_list()
        print(results)
        results.to_csv("results.csv", index = True)
    """
    
    ids_with_good_data = [id for id in ids if not id in [1201, 1202, 1283]]
    res = pipeline(system_ids = ids_with_good_data,
                    training_features = features,
                    target_feature = F.PVDAQ_DC_POWER,
                    clip_features = {F.PVDAQ_DC_POWER: (0, None)},
                    filter_features = {F.PVDAQ_DC_POWER: (0, 3000)},
                    ml_model = ml_model,
                    file_limit = None, mute_tqdm = False,
                    training_data_cache = "training_data.parquet",
                    hyper_parameter_search = False,
                    use_cached_training_data=True)