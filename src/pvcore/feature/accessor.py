from __future__ import annotations
from typing import Dict, Any, Union

from pandas.api.extensions import register_dataframe_accessor
import pandas as pd
import numpy as np
from pathlib import Path
import os

from .catalog import Source, Feature, FEATURE_FROM_NAME, ALL_FEATURE_NAMES
from .processing import Processing as fp
import pvcore.utils.file_utilities as fu

# alias for typing, allowing a single or a list of features
FeatureList = Union[Feature, tuple[Feature], list[Feature], None] 

show_warnings = os.environ.get("FEATURE_DEBUG", "False") == "True"

@register_dataframe_accessor("ftr")
class Accessor:
    """
    Create for every pd.DataFrame df a pandas accessor df.ftr
    for convenient access and management of features, relevant constants and metadata.
    """
    def __init__(self, pandas_obj: Union[pd.DataFrame, pd.Series]):
        self._df: pd.DataFrame = pandas_obj
        self._constants: Dict[Feature, Any] = {}

    def available(self, feature: Feature) -> bool:
        return feature.name in self._df.columns.tolist() + [self._df.index.name]
    
    @property
    def features(self) -> list[Feature]:
        return [FEATURE_FROM_NAME[col] for col in self._df.columns.tolist() + [self._df.index.name] if col in ALL_FEATURE_NAMES]

    def get(self, feature: FeatureList = None) -> Union[pd.Series, pd.DataFrame]:
        """
        Return given features as a pd.Series (or as a constant)
        Read from given pf.DataFrame or calculates it missing.
        """
        if feature is None:
            feature = self.features
        if isinstance(feature, list) or isinstance(feature, tuple):
            missing_features = [ftr for ftr in feature if not self.available(ftr)]
            for ftr in missing_features:
                self.get(ftr)
            df = self._df[[ftr.name for ftr in feature if ftr.name != self._df.index.name]]
            if any(self._df.index.name == ftr.name for ftr in feature):
                df.insert(0, self._df.index.name, self._df.index)
            df.ftr.set_const(self._constants.copy())
            return df
        if self.available(feature):
            if feature.name == self._df.index.name:
                return self._df.index.to_series()
            return self._df[feature.name]
        if feature.is_constant:
            series = pd.Series(data = self.get_const(feature), index = self._df.index)
        else:
            series = self._calculate(feature)
        self._df[feature.name] = series
        return series
    
    def set(self, feature: Feature, col: pd.Series):
        if len(col) != len(self._df):
            raise ValueError(f"Series {col} must have the same length as the DataFrame!")
        col.index = self._df.index
        self._df.loc[:, feature.name] = col

    def get_const(self, const_feature: FeatureList = None) -> Union[float, Dict[Feature, np.dtype]]:
        """
        Return given constant; if constant has not yet been calculated before,
        delegate the calculation and save its result.
        """
        if const_feature is None:
            return self._constants
        if isinstance(const_feature, list):
            return {ftr: self.get_const(ftr) for ftr in const_feature}
        if const_feature in self._constants.keys():
            return self._constants[const_feature]
        result = self._calculate(const_feature)
        self.set_const({const_feature: result})
        return result
    
    def set_const(self, value_dict: dict[Feature, Any]):
        self._constants.update(value_dict)

    def _calculate(self, feature: Feature, show_warnings: bool = show_warnings) -> Union[pd.Series, np.dtype]:
        """Wrapper method for error handling and potential type casting when calculating missing features"""
        if feature.source != Source.CALCULATED:
            if show_warnings:
                print(f"Warning: Cannot calculate feature {feature.name}. Must be loaded from {feature.source.value}.")
            if feature.is_constant:
                return np.nan
            return pd.Series(np.nan, index=self._df.index)
        if feature.required_features is not None:
            const_list = self.get_const([ftr for ftr in feature.required_features if ftr.is_constant])
            df = self.get([ftr for ftr in feature.required_features if not ftr.is_constant])
            nan_cols = [col for col in df.columns if df[col].isna().all().any()]
            nan_consts = [const for const in const_list if pd.isna(const)]
            if nan_cols or nan_consts:
                if show_warnings:
                    print(f"Warning: {nan_cols + nan_consts} is missing to calculate feature {feature}.")
                if feature.is_constant:
                    return np.nan
                return pd.Series(np.nan, index=self._df.index, dtype=feature.data_type)
        try:
            result = fp.calculate(feature = feature, api = self)
        except NotImplementedError:
            print(f"Error: Calculation of feature {feature.name} is not implemented.")
            if feature.is_constant:
                return np.nan
            return pd.Series(np.nan, index=self._df.index, dtype=feature.data_type)
        if feature.is_constant:
            return feature.data_type(result)
        #return result.astype(feature.data_type)
        return result
    
    def drop(self, feature: FeatureList) -> pd.DataFrame:
        """Remove column(s) corresponding to given feature(s) if they are available"""
        if feature is None:
            feature = [ftr for ftr in self.features if ftr.name != self._df.index.name]
        if isinstance(feature, Feature):
            feature = [feature]
        self._df = self._df.drop(columns = [ftr.name for ftr in feature], errors="ignore")
        return self._df

    def copy(self, feature: FeatureList | None = None) -> pd.DataFrame:
        """Returns a copied dataframe with the requested features and same constants."""
        df = self.get(feature).copy()
        df.ftr.set_const(self._constants.copy())
        return df

    def to_csv(self, path: Path, feature: FeatureList | None = None, index = True):
        path = fu.absolute_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if feature is None:
            feature = [ftr for ftr in self.features if ftr.name != self._df.index.name]
        self.get(feature).to_csv(path, index = index)
        
    def filter(self, min_max_dict: dict) -> pd.DataFrame:
        """
        Filter the DataFrame by min/max values for given features.
        (To filter only by one bound, provide None for the other)
        Remove rows out of bound or containing None/NaN values.
        """
        for feature in min_max_dict.keys():
            if not self.available(feature):
                self.get(feature)
        for feature, (min_value, max_value) in min_max_dict.items():
            if min_value is None:
                self._df = self._df[self._df[feature.name] <= max_value]
            elif max_value is None:
                self._df = self._df[self._df[feature.name] >= min_value]
            else:
                self._df = self._df[(self._df[feature.name] >= min_value) & (self._df[feature.name] <= max_value)]
        return self._df
    
    def clip(self, min_max_dict: dict) -> pd.DataFrame:
        """
        Filter the DataFrame by min/max values for given features.
        (To filter only by one bound, provide None for the other)
        Round row out of bound to min/max values.
        """
        for feature in min_max_dict.keys():
            if not self.available(feature):
                self.get(feature)
        for feature, (min_value, max_value) in min_max_dict.items():
            self.set(feature, self.get(feature).clip(lower = min_value, upper = max_value))
        return self._df
 
    def dropna(self, feature: FeatureList = None, how = "any") -> pd.DataFrame:
        if feature is None:
            feature = [ftr for ftr in self.features if ftr.name != self._df.index.name]
        if isinstance(feature, Feature):
            feature = [feature]
        self._df = self._df.dropna(subset = [ftr.name for ftr in feature], how = how)
        return self._df