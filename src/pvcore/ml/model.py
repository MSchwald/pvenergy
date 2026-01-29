from __future__ import annotations
from typing import Any

import pandas as pd
import sklearn.preprocessing
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import lightgbm as lgb
import optuna
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
from optuna.integration import OptunaSearchCV
from optuna.samplers import TPESampler
    
from pvcore.feature import FEATURE_FROM_NAME
from pvcore.paths import MODELS_DIR
from .evaluation import EVALUATIONS, ALL_EVALUATIONS

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
    estimator: Any
    scaler: Any = None
    evaluation_methods: tuple[str, ...] = (EVALUATIONS.RMSE.name, EVALUATIONS.R2.name, EVALUATIONS.FEATURE_IMPORTANCE.name)
    # search for best hyperparmeters with RandomizedSearchCV
    hyperparameters: dict | None = None # possible hyperparam combinations to choose from
    n_iter_search: int = 15 # amount of random combinations to compare
    # trained model gets saved here for further use
    _trained_model: Any = None
    _fitted_scaler: Any = None
    _training_features: tuple[str, ...] | None = None
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

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        # For some models specific rescaling of the training data is important for their performance
        X_train_scaled = self.apply_scaler(X_train, train = True)
        model = self.estimator

        model.n_jobs = -1
        model.random_state = 42
        model.fit(X_train_scaled, y_train)
        self._trained_model = model
        self._training_features = tuple(X_train.columns)
        self._target_feature = str(y_train.name)
        return self._trained_model

    def tune(self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int = 10,
        cv: int = 3
    ) -> dict:
        if self.hyperparameters is None:
            return {}
        if self.scaler is not None:
            pipeline = Pipeline([
                ("scaler", self.scaler()),
                ("model", self.estimator)
            ])
            param_space = {f"model__{k}": v for k, v in self.hyperparameters.items()}
        else:
            pipeline = self.estimator
            param_space = self.hyperparameters
        idx = X_train.index
        assert isinstance(idx, pd.DatetimeIndex)
        groups = idx.normalize()
        cv_splitter = GroupKFold(n_splits=cv)
        storage_url = "sqlite:///optuna.db"
        study_name = f"{self.name}_study2"
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="maximize",
            sampler=sampler,
            load_if_exists=True,
        )
        search = OptunaSearchCV(
            estimator=pipeline,
            param_distributions=param_space,
            n_trials=n_trials,
            scoring="neg_root_mean_squared_error",
            cv=cv_splitter,
            n_jobs=-1,
            study=study,
            refit=True,
            verbose=1,
        )
        search.fit(X_train, y_train, groups=groups.values)
        self._trained_model = search.best_estimator_
        self._training_features = tuple(X_train.columns)
        self._target_feature = str(y_train.name)

        print("Best params:", search.best_params_)
        print("Best CV RMSE:", -search.best_score_)

        return search.best_params_

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if self._trained_model is None or self._training_features is None:
            raise RuntimeError(f"Model {self.name} has not been trained yet.")
        features = tuple(FEATURE_FROM_NAME[name] for name in self._training_features)
        X = self.apply_scaler(X_test.ftr.get(features)) # type: ignore
        return pd.Series(self._trained_model.predict(X), index = X.index)

    def evaluate(self, X_test, y_test, y_pred) -> pd.Series:
        result_list: list[pd.Series] = []
        for method in ALL_EVALUATIONS:
            if method.name in self.evaluation_methods:
                result_list.append(method.evaluate(self._trained_model, X_test, y_test, y_pred))
        results: pd.Series = pd.concat(result_list)
        self._evaluation_results = results
        return results

    def get_hyperparameters(self) -> dict[str, Any]:
        default_params = self.estimator.__class__().get_params()
        current_params = self.estimator.get_params()
        return {
            param: value for param, value in current_params.items()
            if param in default_params and value != default_params[param]
        }

    def save(self, file_name: str) -> None:
        path = MODELS_DIR / f"{file_name}.joblib"
        path.parent.mkdir(parents = True, exist_ok = True)
        joblib.dump(self, path, compress = 3)

    @classmethod
    def load(cls, file_name: str) -> Model:
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
        hyperparameters = {
            'n_estimators': IntDistribution(50, 400),
            'max_depth': IntDistribution(1, 20),
            'min_samples_split': IntDistribution(2, 20),
            'min_samples_leaf': IntDistribution(2, 20),
            'max_features': CategoricalDistribution(["sqrt", "log2", 0.3, 0.5, 0.8])
        },
        n_iter_search = 8
    )
    
    XGBOOST = Model(
        name = "xgboost",
        estimator = XGBRegressor(
            n_estimators=406,
            max_depth=14,
            min_child_weight=8,
            learning_rate=0.035512741148809444,
            subsample=0.8583962991871534,
            colsample_bytree=0.9549411962727286,
            gamma=2.5040366080163086,
            reg_lambda=4.900705776865554,
            reg_alpha=0.19168734368354134,
            #objective='reg:tweedie',
            #tweedie_variance_power=1.5
        ),
        hyperparameters = {
            "n_estimators": IntDistribution(200, 450),
            "learning_rate": FloatDistribution(0.03, 0.035, log=True),
            #"max_depth": IntDistribution(11, 16),
            "min_child_weight": IntDistribution(4, 9),
            "subsample": FloatDistribution(0.83, 0.86),
            "colsample_bytree": FloatDistribution(0.95, 0.965),
            "gamma": FloatDistribution(1.55, 1.91),
            "reg_alpha": FloatDistribution(0.12, 0.38, log=True),
            "reg_lambda": FloatDistribution(0.9, 30.0, log=True),
        }
    )
    LIGHTGBM = Model(
        name = "lightgbm",
        estimator = lgb.LGBMRegressor(
            n_estimators=247,
            max_depth=12,
            learning_rate=0.06914280327719333,
            subsample=0.9275116444792041,
            subsample_freq=2,
            colsample_bytree=0.9723641515413798,
            num_leaves = 141,
            min_child_samples = 54,
            reg_alpha = 5.662928814576938,
            reg_lambda = 0.05574435732973955
        ),
        hyperparameters = {
            'n_estimators': IntDistribution(100, 250),
            'max_depth': IntDistribution(11, 13),
            'learning_rate': FloatDistribution(0.04, 0.3, log=True),
            'subsample': FloatDistribution(0.8, 1),
            'subsample_freq': IntDistribution(1, 5),
            'colsample_bytree': FloatDistribution(0.85, 1.0),
            'num_leaves': IntDistribution(80, 150),
            'min_child_samples': IntDistribution(50, 150),
            'reg_alpha': FloatDistribution(0.001, 10.0, log=True),
            'reg_lambda': FloatDistribution(0.001, 10.0, log=True)
        }
    )