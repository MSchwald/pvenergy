from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from feature.catalog import Feature, FeatureList

import pyarrow.dataset as ds
import pandas as pd

from pvcore.paths import PVDAQ_DIR
from pvcore.feature import Catalog as F
from pvcore.utils import file_utilities as fu

class Pvdaq:
    """Request pv data and metadata of PVDAQ systems."""

    url = "s3://oedi-data-lake/pvdaq"
    META_COLUMN_NAME_MAP = {
        "system_id": F.SYSTEM_ID.name,
        "azimuth": F.AZIMUTH.name,
        "tilt": F.TILT.name,
        "latitude": F.LATITUDE.name,
        "longitude": F.LONGITUDE.name,
        "measured_on": F.TIME.name
    }
    DATA_COLUMN_PREFIX_MAP = {
        "dc_power": F.PVDAQ_DC_POWER,
        "module_temp": F.PVDAQ_MODULE_TEMP,
        "poa_irradiance": F.PVDAQ_POA_IRRADIANCE
    }
    DATA_COLUMN_PREFIXES = list(DATA_COLUMN_PREFIX_MAP.keys())
    DATA_COLUMNS = list(DATA_COLUMN_PREFIX_MAP.values())
    DATA_COLUMNS_NAMES = [ftr.name for ftr in DATA_COLUMNS]

    METADATA_FILE = PVDAQ_DIR / "metadata.csv"
    METRIC_FILE = PVDAQ_DIR / "metric_ids.csv"

    #Cache for PVDAQ metadata and system constants
    _metadata = None
    _system_ids = None
    _metrics = None
    _metric_ids = None

    @classmethod
    def get_metadata(cls) -> pd.DataFrame:
        """Load metadata like location, area, orientation
        for all IDs of pvdaq systems"""
        if cls._metadata is not None:
            return cls._metadata
        if cls.METADATA_FILE.exists():
            metadata_df = pd.read_csv(cls.METADATA_FILE)
            cls._metadata = metadata_df.ftr.get().set_index(F.SYSTEM_ID.name)
            return cls._metadata

        print("Loading PVDAQ metadata")
        prefix = cls.url + "/parquet/"
        data = {}
        #load metadata files
        for directory in ("mount", "site", "system"):
            data[directory] = fu.concat_files(prefix + directory, "parquet", mute_tqdm = True)

        # merge metadata files
        metadata_df = (
            data["system"]
            .merge(data["site"], on="system_id", how="outer")
            .merge(data["mount"], on="system_id", how="outer")
        )
        metadata_df = metadata_df.rename(columns = cls.META_COLUMN_NAME_MAP)
        metadata_df = metadata_df.ftr.get().set_index(F.SYSTEM_ID.name)

        for id in [id for id in metadata_df.index if id != 4901]: # System no. 4901 has two rows of metadata
            for ftr in (F.LATITUDE, F.LONGITUDE): # For some systems the GPS coordinates are off by a factor 1000
                val = metadata_df.loc[id, ftr.name]
                if pd.notna(val) and abs(float(val)) >= 1000:
                    metadata_df.loc[id, ftr.name] = float(val) / 1000
        metadata_df.to_csv(cls.METADATA_FILE, index = True)

        cls._metadata = metadata_df
        return cls._metadata      

    @classmethod
    def meta(cls, system_id: int) -> dict[Feature, Any]:
        """Shortcut to obtain the metadata of a single system as a dictionary."""
        meta_df = cls.get_metadata().reset_index()
        row = meta_df[meta_df[F.SYSTEM_ID.name] == system_id].squeeze().to_dict()
        return {feature: row[feature.name] for feature in meta_df.ftr.features} 

    @classmethod
    def get_system_ids(cls) -> tuple[int]:
        if cls._system_ids is not None:
            return cls._system_ids
        return tuple(cls.get_metadata().index)
    
    @classmethod
    def get_good_data_system_ids(cls) -> tuple[int]:
        """System 4901 has two different sets of recorded metadata, hence we prefer to exclude it for now."""
        return tuple(id for id in cls.get_system_ids() if id != 4901)

    @classmethod
    def get_metrics(cls) -> pd.DataFrame:
        """Names, description, units and comments on the metrics recorded by PVDAQ pv systems"""
        if cls._metrics is not None:
            return cls._metrics
        metric_df = fu.concat_files(
            directory = cls.url + "/parquet/metrics",
            file_format ="parquet",
            cache_directory = PVDAQ_DIR
        )
        metric_df.to_csv(cls.METRIC_FILE, index = False)
        ids = cls.get_metadata().index
        cls._metrics = pd.DataFrame(index = ids, columns = cls.DATA_COLUMNS_NAMES)
        cls._metric_ids = pd.DataFrame(index = ids, columns = cls.DATA_COLUMNS_NAMES, dtype = "Int64")
        cls._metrics.index.name = F.SYSTEM_ID.name
        for id in ids:
            for prefix in cls.DATA_COLUMN_PREFIXES:
                df = metric_df[metric_df["system_id"] == id]
                mask = df["standard_name"].str.startswith(prefix)
                if mask.any():
                    row = metric_df.loc[df.index[mask][0]]
                    cls._metrics.at[id, cls.DATA_COLUMN_PREFIX_MAP[prefix].name] = row["standard_name"]
                    cls._metric_ids.at[id, cls.DATA_COLUMN_PREFIX_MAP[prefix].name] = int(row["metric_id"])
        return cls._metrics

    @classmethod
    def get_metric_ids(cls) -> pd.DataFrame:
        """
        Translate the standard column names in PVDAQ csv files to their ids at their end,
        as they are used as column names in PVDAQ parquet format.
        """
        cls.get_metrics()
        return cls._metric_ids

    @classmethod
    def filter_systems(cls, metacols: FeatureList = None) -> list[int]:
        """
        Filter pvdaq systems based on metadata criteria.
        Returns a list of system IDs that meet the criteria.
        """
        df = pd.concat([cls.get_metadata(), cls.get_metrics()], axis = 1)
        df = df.ftr.dropna(metacols, how="any")
        return list(df.index)
    
    @classmethod
    def load_raw_data(cls,
        system_id: int,
        file_format: str = "parquet",
        cache_directory: str | None = PVDAQ_DIR,
        cache_single_files: bool = False,
        use_columns: list | None = None,
        parquet_filter: list[tuple] | None = None,
        file_limit: int | None = None,
        mute_tqdm: bool = False
    ) -> pd.DataFrame:
        """
        Load raw dataset of a pvdaq pv system with a given id for all recorded times.
        number of downloaded files and the set of used columns can be restricted.
        """
        directory = f"{cls.url}/{file_format}/pvdata/system_id={system_id}"
        parquet_pivot = None
        if file_format == "parquet":
            parquet_pivot = (
                'measured_on', 'metric_id', 'value', 'first'
            )
        df = fu.concat_files(
            directory, file_format, cache_directory = cache_directory, cache_single_files = cache_single_files,
            use_columns = use_columns, parquet_filter = parquet_filter, parquet_pivot = parquet_pivot,
            file_limit = file_limit, mute_tqdm = mute_tqdm
        )
        return df
    
    @classmethod
    def load_measured_features(cls,
        system_id: int,
        file_format: str = "parquet",
        cache_directory: str | None = PVDAQ_DIR,
        cache_single_files: bool = False,
        file_limit: int | None = None,
        mute_tqdm: bool = False
    ) -> pd.DataFrame:
        """
        Load dataset of a pvdaq pv system with a given id containing measured DC power data.
        """
        metric_names = cls.get_metrics().loc[system_id].to_dict()
        metric_ids = cls.get_metric_ids().loc[system_id].to_dict()
        measured_features = [ftr for ftr in cls.DATA_COLUMNS if not pd.isna(metric_names[ftr.name])]
        column_rename_map = {metric_names[ftr.name]: ftr.name for ftr in measured_features}
        column_rename_map["measured_on"] = F.TIME.name
        use_columns = list(column_rename_map.keys())
        parquet_filter = None
        if file_format == "parquet":
            # Replace all column names with the ids they end with (except for the time column)
            ids = [metric_ids[ftr.name] for ftr in measured_features]
            new_rename_map = {str(metric_ids[ftr.name]): ftr.name for ftr in measured_features}
            new_rename_map["measured_on"] = F.TIME.name
            column_rename_map = new_rename_map
            use_columns = ['measured_on', 'metric_id', 'value']
            parquet_filter = ds.field("metric_id").isin(ids)
        df = cls.load_raw_data(
            system_id = system_id, file_format = file_format, cache_directory = cache_directory, cache_single_files = cache_single_files,
            use_columns = use_columns, parquet_filter = parquet_filter, file_limit = file_limit, mute_tqdm = mute_tqdm
        ).rename(columns = column_rename_map)
        df = df.set_index(F.TIME.name)
        df.index = pd.to_datetime(df.index)
        df.ftr.set_const(cls.meta(system_id)) # Keep PVDAQ metadata
        return df