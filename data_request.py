from __future__ import annotations
from feature_accessor import FeatureAccessor

from pathlib import Path
import requests
from apidata import NSRDB_API_KEY, EMAIL # personal data not to be shared in repository
from io import StringIO
from tqdm import tqdm

import pyarrow.dataset as ds

from feature_catalog import FeatureCatalog as F
from feature_catalog import Feature

import pandas as pd
from typing import Any, Union
import file_utilities as fu
import re

from timezonefinder import TimezoneFinder
from datetime import datetime, timedelta

FeatureList = Union[Feature, tuple[Feature], list[Feature], None] 

BASE_DIR = Path(__file__).resolve().parent

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

    LOCAL_DIR = BASE_DIR / "pvdata"
    METADATA_FILE = LOCAL_DIR / "metadata.csv"
    METRIC_FILE = LOCAL_DIR / "metric_ids.csv"

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
            metadata_df = pd.read_csv(cls.METADATA_FILE, index_col = F.SYSTEM_ID.name)
            metadata_df = metadata_df.ftr.get()
            cls._metadata = metadata_df
            return metadata_df

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
        metadata_df = metadata_df.set_index(F.SYSTEM_ID.name)

        for id in [id for id in metadata_df.index if id != 4901]: # System no. 4901 has two rows of metadata
            for ftr in (F.LATITUDE, F.LONGITUDE): # For some systems the GPS coordinates are off by a factor 1000
                val = metadata_df.loc[id, ftr.name]
                if pd.notna(val) and abs(float(val)) >= 1000:
                    metadata_df.loc[id, ftr.name] = float(val) / 1000
        metadata_df.to_csv(cls.METADATA_FILE, index = True)

        cls._metadata = metadata_df.ftr.get() # delete non-feature columns
        return cls._metadata      

    @classmethod
    def meta(cls, system_id: int) -> dict[Feature, Any]:
        """Shortcut to obtain the metadata of a single system as a dictionary."""
        meta_df = cls.get_metadata()
        row = meta_df.loc[system_id]
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
            cache_directory = cls.LOCAL_DIR
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
        df = cls.get_metadata()
        df = df.ftr.dropna(metacols, how="any")
        return list(df.index)
    
    @classmethod
    def load_raw_data(cls,
        system_id: int,
        file_format: str = "parquet",
        cache_directory: str | None = LOCAL_DIR,
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
        cache_directory: str | None = LOCAL_DIR,
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

class Nsrdb:
    """Request weather data for given locations, specifically for PVDAQ pv systems."""

    COLUMN_NAME_MAP = {
        "Temperature": F.AIR_TEMP.name,
        "Clearsky DHI": F.NSRDB_CLEAR_SKY_DHI.name,
        "Clearsky DNI": F.NSRDB_CLEAR_SKY_DNI.name,
        "Clearsky GHI": F.NSRDB_CLEAR_SKY_GHI.name,
        "DHI": F.DHI.name,
        "DNI": F.DNI.name,
        "GHI": F.GHI.name,
        "Surface Albedo": F.SURFACE_ALBEDO.name,
        "Wind Speed": F.WIND_SPEED.name,
        "Wind Direction": F.WIND_DIRECTION.name,
        "time": F.TIME.name
    }
    LOCAL_DIR = BASE_DIR / "weatherdata"
    
    @classmethod
    def load_year(cls,
        latitude: float,
        longitude: float,
        year: int,
        save_result: bool = True
    ) -> pd.DataFrame:
        """Download NSRDB weather data for given GPS location and year.
        Saves data in weaterdata/ which gets loaded if the data is
        requested again."""

        output_file = cls.LOCAL_DIR / f"data_lat={latitude},lon={longitude},y={year}.csv"

        # Request data only if not already downloaded
        if output_file.exists():
            df = pd.read_csv(output_file, parse_dates = [F.TIME.name], index_col = F.TIME.name)
            return df.rename(columns = cls.COLUMN_NAME_MAP)

        #output_metafile = cls.LOCAL_DIR / f"meta_lat={latitude},lon={longitude},y={year}.csv"

        # Load all attributes that are relevant to calculate POA irridiance and temperature of a PV system
        attributes = ["air_temperature", "clearsky_dhi", "clearsky_dni", "clearsky_ghi",
                    "dhi", "dni", "ghi", "surface_albedo",                    
                    "wind_direction", "wind_speed"]
        # Other available attributes, less interesting for us:
        #["cloud_fill_flag", "cloud_type", "dew_point","ozone", "relative_humidity","solar_zenith_angle",
        # "ssa","surface_pressure", "total_precipitable_water"]

        url = "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv"
        params = {
            "api_key": NSRDB_API_KEY,
            "wkt": f"POINT({longitude} {latitude})",
            "attributes": ",".join(attributes),
            "names": str(year),
            "utc": "false",
            "leap_day": "true",
            "email": EMAIL
        }

        response = requests.get(url, params = params, timeout = 30)
        if response.status_code != 200:
            print(f"HTTP Error: {response.status_code}")
            print(response.text)
            return
            #raise ValueError("API request failed")

        # NSRDB metadata is so far unused
        #meta = pd.read_csv(StringIO(response.text), nrows = 1)
        data = pd.read_csv(StringIO(response.text), skiprows = 2)
        data.insert(0,
            F.TIME.name,
            pd.to_datetime(
                dict(
                    year=data['Year'],
                    month=data['Month'],
                    day=data['Day'],
                    hour=data['Hour'],
                    minute=data['Minute']
                )
            )
        )
        data = data.drop(columns=['Year','Month','Day','Hour','Minute'])
        data = data.rename(columns = cls.COLUMN_NAME_MAP)
        data = data.set_index(F.TIME.name)
        data.ftr.set_const({F.LATITUDE: latitude, F.LONGITUDE: longitude})

        if save_result:
            #meta.to_csv(output_metafile, index = True)
            data.to_csv(output_file, index = True)
 
        return data

    @classmethod
    def load_time_range(cls,
        latitude: float,
        longitude: float,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        cache_single_files: bool = False,
        mute_tqdm: bool = False
    ) -> pd.DataFrame:
        """Load several years at once and restrict the data to a given time range."""
        start_year = start_date.year
        end_year = end_date.year
        years = list(range(start_year, end_year+1))
        dfs = []
        for year in years if mute_tqdm else tqdm(years, desc=f"Loading weather data from {start_date}-{end_date} - CSVs"):
            if not mute_tqdm:
                tqdm.write(f"Loading weather data from year {year}")
            dfs.append(cls.load_year(latitude, longitude, year, save_result = cache_single_files))
        
        dfs[0] = dfs[0][dfs[0].index >= start_date]
        dfs[-1] = dfs[-1][dfs[-1].index <= end_date]
        df = pd.concat(dfs)
        df.ftr.set_const({F.LATITUDE: latitude, F.LONGITUDE: longitude})

        return df

    @classmethod
    def load_system(cls,
        api: FeatureAccessor,
        cache_directory: str | None = LOCAL_DIR,
        cache_single_files: bool = False,
        mute_tqdm = False
    ) -> pd.DataFrame:
        """Load weather data with calculated global poa for pv system with given system id."""
        id = api.get_const(F.SYSTEM_ID)
        if cache_directory is not None:
            cache_directory.mkdir(parents = True, exist_ok = True)
            cache_path = cls.LOCAL_DIR / f"weather_system_id={id}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        start = api._df.index[api._df.index.year >= 1998].min()
        end = api._df.index[-1]
        data = Nsrdb.load_time_range(
            api.get_const(F.LATITUDE),
            api.get_const(F.LONGITUDE),
            start,
            end,
            cache_single_files = cache_single_files,
            mute_tqdm = mute_tqdm
        )
        if cache_directory is not None:
            data.to_parquet(cache_path, index = True)

        data.ftr.set_const(api.get_const())
        return data

class OpenMeteo:
    """Request live / forecast weather data"""
    url = "https://api.open-meteo.com/v1/forecast"
    COLUMN_NAME_MAP = {
        "time": F.UTC_TIME.name,
        "temperature_2m": F.AIR_TEMP.name,
        "shortwave_radiation": F.GHI.name,
        "diffuse_radiation": F.DHI.name,
        "direct_normal_irradiance": F.DNI.name,
        "wind_speed_10m": F.WIND_SPEED.name,
        "wind_direction_10m": F.WIND_DIRECTION.name,
    }
    LOCAL_DIR = BASE_DIR / "live_weatherdata"
    tf = TimezoneFinder()

    @classmethod
    def get_forecast(cls,
        latitude: float,
        longitude: float,
        save_result: bool = True
    ) -> pd.DataFrame:
        """
        Get hourly weather forecast for given location for the next day.
        Caching is used for multiple requests in the same hour of the day.
        """
        cls.LOCAL_DIR.mkdir(parents = True, exist_ok = True)
        file = cls.LOCAL_DIR / f"openmeteo_lat={latitude}_lon={longitude}.csv"
        if file.exists():
            mtime = datetime.fromtimestamp(file.stat().st_mtime)
            time_now = datetime.now()
            if time_now - mtime < timedelta(hours = 1) and time_now.hour == mtime.hour:
                df = pd.read_csv(file, index_col = F.UTC_TIME.name)
                df.index = pd.to_datetime(df.index)
                print("Cached version of OpenMeteo data")
                return df
        print("request new version from OpenMeteo")
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": [
                "shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance",
                "temperature_2m", "wind_speed_10m", "wind_direction_10m"
            ],
            "models": "gfs_seamless",
            "current_weather": True,
            "forecast_hours": 24
        }
        response = requests.get(cls.url, params=params, timeout=30)
        response.raise_for_status()

        # Stundenweise Daten
        df = pd.DataFrame(data = response.json()["hourly"]).rename(columns=cls.COLUMN_NAME_MAP).set_index(F.UTC_TIME.name)
        df.index = pd.to_datetime(df.index)
        
        if save_result:
            df.to_csv(file, index = True)

        return df

def request_data(
    system_id: int,
    file_limit: int | None = None,
    output_dir: str | None = "requested_data",
    mute_tqdm = False
) -> pd.DataFrame:
    """Requests features from PVDAQ and NSRDB for system with given ID."""
    cache_dir = fu.absolute_path(output_dir)
    cache = cache_dir / f"pv_and_weather_system_id={system_id}.parquet"
    if cache.exists():
        df = fu.get_file(cache)
        df.ftr.set_const(Pvdaq.meta(system_id))
        return df
    pv_data = Pvdaq.load_measured_features(system_id = system_id, file_limit = file_limit, mute_tqdm = mute_tqdm)
    meta = pv_data.ftr.get_const()
    if pv_data.empty:
        return pd.DataFrame()
    weather_data = Nsrdb.load_system(pv_data.ftr, mute_tqdm = mute_tqdm).sort_index()
    pv_data = pv_data.sort_index().reindex(
        weather_data.index
    ).interpolate(
        method="time", limit = 1, limit_area = "inside"
    ).join(weather_data, how='inner')
    if output_dir is not None and file_limit is None:
        #df.to_csv(Path(output_dir) / f"{system_id}.csv", index = True)
        pv_data.to_parquet(cache, index = True)
    pv_data.ftr.set_const(meta)
    return pv_data

def get_features(
        system_id: int,
        features: FeatureList = None,
        file_limit: int | None = None,
        mute_tqdm = False
    ) -> pd.DataFrame:
    """
    Download PVDAQ and NSRDB data for a given system id and calculate a list of given features.
    (Method could later be expanded to automatically adapt data requests precisely to
    only the necessary features. I skipped this for now, as on the other hand,
    downloading all data maximally necessary for the implemented FeatureCatalog and caching
    the results saves further data requests upon changing the provided list of training features.)
    """
    df = request_data(system_id, file_limit = file_limit, mute_tqdm = mute_tqdm)        
    if df.empty:
        print(f"No measured data available for system ID {system_id}.")
        return
    return df.ftr.get(features)

if __name__ == "__main__":
    """Testing space for data requests"""
    #df = Pvdaq.load_raw_data(2, file_format = "csv", file_limit = 10)
    #print(df)
    #df = pd.read_csv("live_weatherdata/openmeteo_lat=39.7214_lon=-105.0972.csv")
    #print(df.info())
    #print(df.ftr.features)
    #meta = Pvdaq.meta(2)
    #lat, lon = meta[F.LATITUDE], meta[F.LONGITUDE]
    #print(OpenMeteo.get_forecast(lat, lon))
    #print(OpenMeteo.system_forecast(2))