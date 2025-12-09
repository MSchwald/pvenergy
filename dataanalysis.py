# data analysis
import pandas as pd
import numpy as np

# machine learning
import sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import lightgbm as lgb

# modules for working with features
import data_request
from data_request import request_data, Pvdaq, OpenMeteo
from feature_catalog import FeatureCatalog as F
from feature_catalog import Feature
from feature_processing import FeatureProcessing as fp

# python basics
import sys
from pathlib import Path
from typing import Union, Callable, Any
from tqdm import tqdm
from dataclasses import dataclass
import joblib
import file_utilities as fu

FeatureList = Union[Feature, tuple[Feature], list[Feature], None] 

BASE_DIR = Path(__file__).resolve().parent

class Scaler:
    STANDARD = sklearn.preprocessing.StandardScaler
    MINMAX = sklearn.preprocessing.MinMaxScaler
    ROBUST = sklearn.preprocessing.RobustScaler
    MAXABS = sklearn.preprocessing.MaxAbsScaler

@dataclass
class EvaluationMethod:
    """Template for creating evaluation methods for trained ML models"""
    name: str
    method: Callable[tuple, pd.Series]
    _result: pd.Series = None

    def evaluate(self, model, X_test, y_test, y_pred):
        result = self.method(model, X_test, y_test, y_pred)
        if isinstance(result, pd.Series):
            self._result = result
        else:
            self._result = pd.Series([result], index = [self.name])
        return self._result
    def print_result(self):
        if self._result is None:
            print(f"Method {self.name} has not been evaluated on test data yet.")
            return 
        print(self.name, self._result)
        
class EVALUATIONS:
    """Methods to analyze the performance of trained ML models"""
    def rmse_method(model, X_test, y_test, y_pred) -> pd.Series:
        return np.sqrt(mean_squared_error(y_test, y_pred))
                       
    def r2_method(model, X_test, y_test, y_pred) -> pd.Series:
        return r2_score(y_test, y_pred)
        
    def feature_importance_method(model, X_test, y_test, y_pred) -> pd.Series:
        df = pd.DataFrame({
            'Feature': X_test.columns.tolist(),
            'Importance': model.feature_importances_
        }).sort_values(by = 'Importance', ascending = False)
        return pd.Series(df['Importance'].values, index = df['Feature']) 

    RMSE = EvaluationMethod(
        name = "rmse",
        method = rmse_method     
    )
    R2 = EvaluationMethod(
        name = "r2",
        method = r2_method
    )
    FEATURE_IMPORTANCE = EvaluationMethod(
        name = "feature_importance",
        method = feature_importance_method
    )
ALL_EVALUATIONS = tuple(eval for eval in vars(EVALUATIONS).values() if isinstance(eval, EvaluationMethod))
    
@dataclass
class Model:
    """Defining properties of ML models"""
    name: str
    estimator: object
    scaler: Scaler | None = None
    evaluation_methods: tuple[str] | None = (EVALUATIONS.RMSE.name, EVALUATIONS.R2.name, EVALUATIONS.FEATURE_IMPORTANCE.name)
    # search for best hyperparmeters with RandomizedSearchCV
    hyperparam_grid: dict | None = None # possible hyperparam combinations to choose from
    n_iter_search: int = 15 # amount of random combinations to compare
    # trained model gets saved here for further use
    _trained_model: object | None = None
    _fitted_scaler: object | None = None
    _training_features: tuple[str] | None = None
    _target_feature: str | None = None
    _evaluation_results: pd.Series | None = None
    
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

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, hyper_parameter_search: bool = True) -> None:
        # For some models specific rescaling of the training data is important for their performance
        X_train_scaled = self.apply_scaler(X_train, train = True)
        model = self.estimator

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
            model.random_state = 42
            model.fit(X_train_scaled, y_train)
            self._trained_model = model
        self._training_features = tuple(X_train.columns)
        self._target_feature = y_train.name
        return self._trained_model
    
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if self._trained_model is None:
            raise RuntimeError(f"Model {self.name} has not been trained yet.")
        features = tuple(fp.FEATURE_FROM_NAME[name] for name in self._training_features)
        X = self.apply_scaler(X_test.ftr.get(features))
        return pd.Series(self._trained_model.predict(X), index = X.index)

    def evaluate(self, X_test, y_test, y_pred):
        result_list = []
        for method in ALL_EVALUATIONS:
            if method.name in self.evaluation_methods:
                method.evaluate(self._trained_model, X_test, y_test, y_pred)
                result_list.append(method._result)
        results = pd.concat(result_list)
        self._evaluation_results = results
        return results

    def save(self, file_name: str):
        path = BASE_DIR / "trained_models" / f"{file_name}.joblib"
        path.parent.mkdir(parents = True, exist_ok = True)
        joblib.dump(self, path, compress = 3)

    @classmethod
    def load(cls, file_name: str):
        if '__main__' in sys.modules:
            sys.modules['__main__'].Model = cls
        path = BASE_DIR / "trained_models" / f"{file_name}.joblib"
        return joblib.load(path)

class ML_MODELS:
    """Collection ML models suitable for analyzing PVDAQ data"""
    RANDOM_FOREST = Model(
        name = "random_forest",
        estimator = RandomForestRegressor(
            n_estimators=200,
            min_samples_split=10,
            min_samples_leaf=2,
            max_features=0.5,
            max_depth=20
        ),
        hyperparam_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.5]
        },
        n_iter_search = 8
    )
    XGBOOST = Model(
        name = "xgboost",
        estimator = XGBRegressor(
            n_estimators=800,
            max_depth=11,
            min_child_weight=3,
            learning_rate=0.12,
            subsample=0.875,
            colsample_bytree=0.96,
            gamma=1.3,
            reg_lambda=0.9,
            reg_alpha=0.1
            #objective='reg:tweedie',
            #tweedie_variance_power=1.5
        ),
        hyperparam_grid = {
            'n_estimators': [700, 800, 900],
            'learning_rate': [0.11, 0.12, 0.13],
            'max_depth': [10, 11, 12],
            'min_child_weight': [3, 4, 5],
            'subsample': [0.825, 0.85, 0.875],
            'colsample_bytree': [0.96, 0.98, 1.0],
            'gamma': [1.1, 1.2, 1.3],
            'reg_alpha': [0.05, 0.1, 0.15],
            'reg_lambda': [0.7, 0.8, 0.9]
            #'tweedie_variance_power': [1.2, 1.5, 1.8]
        }
    )
    LIGHTGBM = Model(
        name = "lightgbm",
        estimator = lgb.LGBMRegressor(
            n_estimators=400,
            max_depth=12,
            learning_rate=0.35,
            subsample=0.95,
            colsample_bytree=1.0,
            num_leaves = 70,
        ),
        hyperparam_grid = {
            'n_estimators': [350, 400, 450],
            'max_depth': [12, 14, 16],
            'learning_rate': [0.3, 0.35, 0.4],
            'subsample': [0.85, 0.9, 0.95],
            'colsample_bytree': [0.85, 0.9, 0.95, 1.0],
            'num_leaves': [65, 70, 75]
        }

    )

class Pipeline:

    _system_constants: pd.DataFrame | None = None
    TRAINING_IDS = [
        id for id in Pvdaq.filter_systems(
            metacols = [F.PVDAQ_DC_POWER, F.PVDAQ_MODULE_TEMP, F.TILT, F.AZIMUTH]
        ) if id in Pvdaq.get_good_data_system_ids() and not id in (50, 51, 1200, 1201, 1202, 1203, 1204, 1283, 1403, 1420) and id < 1422
    ]

    @classmethod
    def get_training_data(cls,
        system_ids: list[int],
        features: list[Feature],
        clip_features: dict[Feature, tuple[int | None, int | None]] = {},
        filter_features: dict[Feature, tuple[int | None, int | None]] = {},
        file_limit: int | None = None, # Safety option and for quick testing pipeline with few data
        mute_tqdm = False, # Reduce info on data processing printed by the console
        training_data_cache: str | None = None, # Cache training data before machine learning
        use_cached_training_data: bool = True
    ) -> pd.DataFrame:

        if training_data_cache is not None and use_cached_training_data:
            training_data_cache = fu.absolute_path(training_data_cache)
            if training_data_cache.exists():
                return pd.read_parquet(training_data_cache)
            #df_full.set_index(F.TIME.name)
        else:
            # Use some already calculated constant features to save time
            system_constants = cls.get_system_constants().dropna(how = "any")
            cached_ids = system_constants.index.to_list()

            # Download pv and weather data and calculate requested features for the given pv systems
            dfs = []
            for system_id in tqdm(system_ids, desc=f"Loading PVDAQ and NSRDB data"):
                tqdm.write(f"Loading data for system {system_id}")
                df = data_request.request_data(system_id = system_id, file_limit = file_limit, mute_tqdm = mute_tqdm)
                if system_id in cached_ids:
                    df.ftr.set_const(cls.system_constants(system_id))
                df = df.ftr.get(features)
                # Clean up data for each system individually
                df = df.ftr.clip(clip_features)
                df = df.ftr.filter(filter_features)
                df = df.ftr.dropna(features)
                if not df.empty:
                    dfs.append(df)
            df_full: pd.DataFrame = pd.concat(dfs)

            if training_data_cache is not None:
                df_full.to_parquet(training_data_cache)
            return df_full

    @classmethod
    def get_system_constants(cls) -> pd.DataFrame:
        """Calculate relevant system constants and cache the result to save time."""
        if cls._system_constants is not None:
            return cls._system_constants
        constant_file = BASE_DIR / "system_constants.csv"
        if constant_file.exists():
            df = pd.read_csv(constant_file, index_col = F.SYSTEM_ID.name)
            cls._system_constants = df
            return df
        meta = Pvdaq.get_metadata()
        META_COLUMNS = [fp.FEATURE_FROM_NAME[col] for col in meta.columns]
        good_ids = cls.TRAINING_IDS
        ids = [id for id in Pvdaq.filter_systems(metacols = [F.PVDAQ_DC_POWER, F.PVDAQ_MODULE_TEMP, F.TILT, F.AZIMUTH]) if id in good_ids]
        constant_features = META_COLUMNS + [F.TIME_ZONE, F.UTC_OFFSET, F.DCP0, F.GAMMA, F.FAIMAN_U0, F.FAIMAN_U1]
        system_constants = pd.DataFrame(data = np.nan, index = ids, columns = [ftr.name for ftr in constant_features])
        system_constants.index.name = F.SYSTEM_ID.name
        for id in tqdm(ids, desc="Calculating system constants..."):
            tqdm.write(f"for system {id}")
            df = request_data(id)
            for ftr in constant_features:
                system_constants.loc[id, ftr.name] = df.ftr.get_const(ftr)
            
        system_constants.to_csv(constant_file, index = True)
        cls._system_constants = system_constants
        return system_constants
    
    @classmethod
    def system_constants(cls, system_id: int) -> dict[Feature, Any]:
        return {fp.FEATURE_FROM_NAME[name]: value for name, value in cls.get_system_constants().loc[system_id].to_dict().items()}

    @staticmethod
    def train_test_split(
        X: pd.DataFrame,
        y: pd.DataFrame,
        test_size: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits data randomly into training and testing data for machine learning"""
        print(f"\nChoosing a random {int(100*(1-test_size))}:{int(100*test_size)} split of the data:")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = test_size,
            random_state = 42 # A random seed, only for reproducibility of the results
        )
        print(f"  {X_train.shape[0]} samples for training data.")
        print(f"  {X_test.shape[0]} samples to testing data.")
        return X_train, X_test, y_train, y_test

    @classmethod
    def weather_forecast(cls, system_id: int) -> pd.DataFrame:
        meta = Pvdaq.meta(system_id)
        df = OpenMeteo.get_forecast(meta[F.LATITUDE], meta[F.LONGITUDE])
        df.ftr.set_const(meta)
        df = cls.utc_to_local_time(df)
        df.ftr.set_const(cls.system_constants(system_id))
        return df

    @classmethod
    def utc_to_local_time(cls, df: pd.DataFrame) -> pd.DataFrame:
        df.reset_index()
        df.insert(
            0,
            F.TIME.name,
            (df.ftr.get(F.UTC_TIME) + pd.Timedelta(
                    hours = df.ftr.get_const(F.UTC_OFFSET)
                )
            ).dt.tz_localize(None)
        )
        df = df.set_index(F.TIME.name)
        return df

    @classmethod
    def predict(cls, model: Model | list[Model], df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(model, Model):
            model = [model]
        untrained = [m for m in model if m._trained_model is None]
        if untrained:
            raise RuntimeError(f"Models {untrained} have not been trained yet.")
        results = []
        sunny = df.ftr.get(F.PVLIB_POA_IRRADIANCE) >= 1
        df_sunny = df.loc[sunny].copy() # can this copy be avoided?
        df_sunny.ftr.set_const(df.ftr.get_const())
        for m in model:
            y = pd.Series(0, index=df.index, dtype=float, name = m.name)
            # Use the ML model to predict when sunny, else return 0
            y.loc[sunny] = m.predict(df_sunny)
            results.append(y)        
        return pd.concat(results, axis = 1)

    @classmethod
    def fleet_analysis(cls,
        system_ids: list[int],
        training_features: list[Feature],
        target_feature: Feature = F.PVDAQ_DC_POWER,
        clip_features: dict[Feature, tuple[int | None, int | None]] = {},
        filter_features: dict[Feature, tuple[int | None, int | None]] = {},
        ml_model: Model = ML_MODELS.RANDOM_FOREST,
        hyper_parameter_search: bool = False,
        file_limit: int | None = None, # Safety option and for quick testing pipeline with few data
        mute_tqdm = False, # Reduce info on data processing printed by the console
        training_data_cache: str | None = None, # Cache training data before machine learning
        use_cached_training_data: bool = True,
        save_model_name: str | None = None
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
            system_ids = system_ids, features = features,
            clip_features = clip_features, filter_features = filter_features,
            file_limit = file_limit, mute_tqdm = mute_tqdm,
            training_data_cache = training_data_cache, use_cached_training_data = use_cached_training_data
        )

        # Split data into training data and target feature
        X: pd.DataFrame = df_full.ftr.get([ftr for ftr in features if ftr != target_feature])
        y: pd.Series = df_full.ftr.get(target_feature)
        
        X_train, X_test, y_train, y_test = cls.train_test_split(X, y)

        print(f"\nTraining ML-model '{ml_model.name}' to predict feature '{target_feature}'.")                      
        ml_model.train(X_train, y_train, hyper_parameter_search)

        print(f"Testing trained model on predicting '{target_feature}' from the randomly chosen testing data.")
        y_pred = ml_model.predict(X_test)

        print("\nAnalysis of the model's performance on the testing data:")
        results = ml_model.evaluate(X_test, y_test, y_pred)
        print(results)
        
        if save_model_name is not None:
            print(f"\nSaving trained model as 'trained_models/{save_model_name}.joblib'...")
            ml_model.save(save_model_name)

        return results
    
    @classmethod
    def individual_analysis(cls, system_ids: list[int],  *args, training_data_cache_dir: str | None = None, **kwargs):
        if training_data_cache_dir is not None:
            local_dir = fu.absolute_path(training_data_cache_dir)
            local_dir.mkdir(exist_ok = True)
        results = []
        for id in system_ids:
            if training_data_cache_dir is not None:
                training_data_cache = local_dir / f"training_data_{id}.parquet"
            else:
                training_data_cache = None
            res = cls.fleet_analysis(system_ids = [id], *args, training_data_cache = training_data_cache, **kwargs)
            res.name = id
            results.append(res)

        df = pd.concat(results, axis = 1).transpose()
        df.index.name = F.SYSTEM_ID.name
        return df

if __name__ == "__main__":
    """Testing space for the pipeline"""
    #print(df.join(Pvdaq.get_metadata()))
    df = Pipeline.weather_forecast(2)
    print(df)
    """print(Pipeline.get_system_constants())
    for id in Pipeline.TRAINING_IDS:
        print(Pvdaq.load_measured_features(id))"""

    #print(Pipeline.get_system_constants())
    # Chooses all system ids where all important features are recorded to predict the default target feature PVDAQ_DC_POWER

    # Current Training features for the ML models
    features: list[Feature] = [
        F.POWER_RATIO, F.PVLIB_POA_IRRADIANCE,
        F.DAY_OF_YEAR, F.TIME_SINCE_SUNLIGHT,
        F.CLEAR_SKY_RATIO, F.COS_AOI, F.WIND_NORMAL_COMPONENT,
        F.POA_COS_AOI, F.POA_WIND_SPEED, F.DHI_PER_GHI,
        F.DCP_PER_AREA, F.GAMMA_TEMP_DIFFERENCE, F.RELATIVE_AZIMUTH
    ]
    
    # Choice of model to train
    #for ml_model in [ML_MODELS.XGBOOST, ML_MODELS.LIGHTGBM, ML_MODELS.RANDOM_FOREST]:
    #ml_model = ML_MODELS.XGBOOST
    #RANDOM_FOREST, XGBOOST, LIGHTGBM
    """    Pipeline.fleet_analysis(
            system_ids = Pipeline.TRAINING_IDS,
            training_features = features,
            target_feature = F.PVDAQ_DC_POWER,
            clip_features = {F.PVDAQ_DC_POWER: (0, None)},
            filter_features = {F.PVDAQ_DC_POWER: (0, 3000), F.PVLIB_POA_IRRADIANCE: (1, None)},
            ml_model = ml_model,
            file_limit = None,
            mute_tqdm = False,
            training_data_cache = "training_data/full_training_data.parquet",
            hyper_parameter_search = False,
            use_cached_training_data = True,
            save_model_name = ml_model.name
        )"""

    """res = Pipeline.individual_analysis(
        system_ids = ids,
        training_features = features,
        target_feature = F.PVDAQ_DC_POWER,
        clip_features = {F.PVDAQ_DC_POWER: (0, None)},
        filter_features = {F.PVDAQ_DC_POWER: (0, 3000), F.PVLIB_POA_IRRADIANCE: (1, None)},
        ml_model = ml_model,
        file_limit = None,
        mute_tqdm = False,
        training_data_cache_dir = "training_data",
        hyper_parameter_search = False,
        use_cached_training_data = True,
        save_model_name = None
    )"""
    #df = Pipeline.weather_forecast(2)
    
    #print(res)
    
    #m = Model.load(ML_MODELS.XGBOOST.name)
    #df = Pipeline.weather_forecast(2)
    #print(df.ftr.get_const())
    #print(Pipeline.predict(m, df))
    #m._training_features = features
    #print([ftr.is_constant for ftr in m._training_features if ftr != F.TIME])
    #print(df.ftr.get_const(F.TIME_ZONE))
    #print(df.ftr.get(lst))
    #print(df.ftr.get([ftr for ftr in m._training_features if ftr != F.TIME]))
    #print(df.ftr.get_const())
    #print(Pipeline.predict(m, df))

    #print(m._evaluation_results)
    #df = pd.read_parquet("training_data/full_training_data.parquet")
    #print(df.info())
    #df = df.ftr.filter({F.PVDAQ_DC_POWER: (0, 3000), F.PVLIB_POA_IRRADIANCE: (1, None)})
    #df.to_csv("training2.csv")