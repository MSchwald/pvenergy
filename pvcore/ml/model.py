import pandas as pd

import sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

from pvcore.feature import FEATURE_FROM_NAME
from pvcore.paths import MODELS_DIR
from .evaluation import EVALUATIONS, ALL_EVALUATIONS

import sys
from dataclasses import dataclass
import joblib

class Scaler:
    STANDARD = sklearn.preprocessing.StandardScaler
    MINMAX = sklearn.preprocessing.MinMaxScaler
    ROBUST = sklearn.preprocessing.RobustScaler
    MAXABS = sklearn.preprocessing.MaxAbsScaler

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
        features = tuple(FEATURE_FROM_NAME[name] for name in self._training_features)
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
    
    def get_hyperparameters(self):
        default_params = self.estimator.__class__().get_params()
        current_params = self.estimator.get_params()
        return {
            param: value for param, value in current_params.items()
            if param in default_params and value != default_params[param]
        }

    def save(self, file_name: str):
        path = MODELS_DIR / f"{file_name}.joblib"
        path.parent.mkdir(parents = True, exist_ok = True)
        joblib.dump(self, path, compress = 3)

    @classmethod
    def load(cls, file_name: str):
        if '__main__' in sys.modules:
            sys.modules['__main__'].Model = cls
        path = MODELS_DIR / f"{file_name}.joblib"
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