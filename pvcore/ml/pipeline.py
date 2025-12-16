from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from feature.catalog import Feature

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from pvcore.io import Pvdaq, Nsrdb, OpenMeteo
from pvcore.feature import Catalog as F, FEATURE_FROM_NAME
import pvcore.utils.file_utilities as fu
from pvcore.paths import MERGED_DIR, TRAINING_DIR, RESULTS_DIR
from .model import Model, ML_MODELS

class Pipeline:

    _system_constants: pd.DataFrame | None = None
    TRAINING_IDS: tuple[int] = tuple(
        id for id in Pvdaq.filter_systems(
            metacols = [F.PVDAQ_DC_POWER, F.PVDAQ_MODULE_TEMP, F.TILT, F.AZIMUTH]
        ) if id in Pvdaq.get_good_data_system_ids() and not id in (50, 51, 1200, 1201, 1202, 1203, 1204, 1283, 1403, 1420) and id < 1422
    )

    def request_data(
        system_id: int,
        cache_name: str | None = "pv_and_weather",
        mute_tqdm = False
    ) -> pd.DataFrame:
        """Requests features from PVDAQ and NSRDB for system with given ID."""
        if cache_name:
            cache = MERGED_DIR / f"{cache_name}_system_id={system_id}.parquet"
            if cache.exists():
                df = fu.get_file(cache)
                df.ftr.set_const(Pvdaq.meta(system_id))
                return df
        pv_data = Pvdaq.load_measured_features(system_id = system_id, mute_tqdm = mute_tqdm)
        meta = pv_data.ftr.get_const()
        if pv_data.empty:
            return pd.DataFrame()
        weather_data = Nsrdb.load_system(pv_data.ftr, mute_tqdm = mute_tqdm).sort_index()
        if not mute_tqdm:
            tqdm.write("Merging PV and weather data...")
        pv_data = pv_data.sort_index().reindex(
            weather_data.index
        ).interpolate(
            method="time", limit = 1, limit_area = "inside"
        ).join(weather_data, how='inner')
        if cache_name is not None:
            pv_data.to_parquet(cache, index = True)
        pv_data.ftr.set_const(meta)
        return pv_data

    @classmethod
    def get_training_data(cls,
        system_ids: tuple[int],
        features: tuple[Feature],
        clip_features: dict[Feature, tuple[int | None, int | None]] = {},
        filter_features: dict[Feature, tuple[int | None, int | None]] = {},
        mute_tqdm = False, # Reduce info on data processing printed by the console
        cache_name: str | None = None, # Cache training data before machine learning
        use_cache: bool = True
    ) -> pd.DataFrame:
        if cache_name is not None:
            cache = TRAINING_DIR / f"{cache_name}.parquet"
            cache_info = TRAINING_DIR / f"{cache_name}.json"
            if use_cache and cache.exists() and cache_info.exists():
                with open(cache_info) as f:
                    info = json.load(f)
                if set(info["system_ids"]) == set(system_ids) and set(info["features"]) == set(ftr.name for ftr in features):
                    return pd.read_parquet(cache)
        # Use some already calculated constant features to save time
        system_constants = cls.get_system_constants().dropna(how = "any")
        cached_ids = system_constants.index.to_list()

        # Download pv and weather data and calculate requested features for the given pv systems
        dfs = []
        for system_id in tqdm(system_ids, desc=f"Preprocessing PVDAQ and NSRDB data"):
            tqdm.write(f"Loading training data for system {system_id}")
            df = cls.request_data(system_id = system_id, mute_tqdm = mute_tqdm)
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

        if cache_name is not None:
            df_full.to_parquet(cache)
            df_info = {
                "system_ids": system_ids,
                "features": [ftr.name for ftr in features]
            }
            with open(cache_info, "w") as f:
                json.dump(df_info, f, indent=2)
        return df_full

    @classmethod
    def get_system_constants(cls) -> pd.DataFrame:
        """Calculate relevant system constants and cache the result to save time."""
        if cls._system_constants is not None:
            return cls._system_constants
        constant_file = RESULTS_DIR / "system_constants.csv"
        if constant_file.exists():
            df = pd.read_csv(constant_file, index_col = F.SYSTEM_ID.name)
            cls._system_constants = df
            return df
        meta = Pvdaq.get_metadata()
        META_COLUMNS = [FEATURE_FROM_NAME[col] for col in meta.columns]
        good_ids = cls.TRAINING_IDS
        ids = [id for id in Pvdaq.filter_systems(metacols = [F.PVDAQ_DC_POWER, F.PVDAQ_MODULE_TEMP, F.TILT, F.AZIMUTH]) if id in good_ids]
        constant_features = META_COLUMNS + [F.TIME_ZONE, F.UTC_OFFSET, F.DCP0, F.GAMMA, F.FAIMAN_U0, F.FAIMAN_U1]
        system_constants = pd.DataFrame(data = np.nan, index = ids, columns = [ftr.name for ftr in constant_features])
        system_constants[F.TIME_ZONE.name] = system_constants[F.TIME_ZONE.name].astype('object')
        system_constants.index.name = F.SYSTEM_ID.name
        for id in tqdm(ids, desc="Requesting raw data from PVDAQ and NSRDB and calculating system constants..."):
            tqdm.write(f"for system {id}")
            df = cls.request_data(id)
            tqdm.write("Calculating system constants...")
            for ftr in constant_features:
                system_constants.loc[id, ftr.name] = df.ftr.get_const(ftr)
            
        system_constants.to_csv(constant_file, index = True)
        cls._system_constants = system_constants
        return system_constants
    
    @classmethod
    def system_constants(cls, system_id: int) -> dict[Feature, Any]:
        return {FEATURE_FROM_NAME[name]: value for name, value in cls.get_system_constants().loc[system_id].to_dict().items()}

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
    def fleet_analysis(cls,
        training_features: tuple[Feature],
        target_feature: Feature = F.PVDAQ_DC_POWER,
        system_ids: tuple[int] = TRAINING_IDS,
        clip_features: dict[Feature, tuple[int | None, int | None]] = {F.PVDAQ_DC_POWER: (0, None)},
        filter_features: dict[Feature, tuple[int | None, int | None]] = {F.PVDAQ_DC_POWER: (0, 3000), F.PVLIB_POA_IRRADIANCE: (1, None)},
        ml_model: Model = ML_MODELS.RANDOM_FOREST,
        hyper_parameter_search: bool = False,
        mute_tqdm = False, # Reduce info on data processing printed by the console
        training_data_cache: str | None = "full_training_data", # Cache training data before machine learning
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
        # Prepare tuple of all requested features
        if target_feature not in training_features:
            features = tuple([target_feature, *training_features])
        else:
            features = training_features

        df_full = cls.get_training_data(
            system_ids = system_ids, features = features,
            clip_features = clip_features, filter_features = filter_features,
            mute_tqdm = mute_tqdm, cache_name = training_data_cache, use_cache = use_cached_training_data
        )

        print([ftr for ftr in features if not df_full.ftr.available(ftr)])

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
    def individual_analysis(cls,
            system_ids: tuple[int] = TRAINING_IDS,
            *args,
            cache_name: str | None = "training_data",
            **kwargs
        ):
        results = []
        for id in system_ids:
            if cache_name is not None:
                training_data_cache = f"{cache_name}_{id}"
            else:
                training_data_cache = None
            res = cls.fleet_analysis(system_ids = (id,), *args, training_data_cache = training_data_cache, **kwargs)
            res.name = id
            results.append(res)

        df = pd.concat(results, axis = 1).transpose()
        df.index.name = F.SYSTEM_ID.name
        return df
    
    @classmethod
    def system_evaluations(cls,
        trained_model: Model,
        system_ids: tuple[int] = TRAINING_IDS,
    ) -> pd.DataFrame:
        features = tuple(FEATURE_FROM_NAME[name] for name in trained_model._training_features)
        if features is None:
            raise RuntimeError(f"Model {trained_model} has not been trained yet.")
        return cls.individual_analysis(training_features = features, system_ids = system_ids, ml_model = trained_model)

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
    def integrate_timeseries(lcs, series: pd.Series) -> float:
        """Numeric integration of a pandas DatetimeIndex via trapezoid rule."""
        dt = series.index.to_series().diff().dt.total_seconds().fillna(0)
        return ((series + series.shift(1)) / 2 * dt / 3600000).sum()

    @classmethod
    def predict(cls, model: Model | tuple[Model], df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(model, Model):
            model = tuple(model,)
        untrained = tuple(m for m in model if m._trained_model is None)
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