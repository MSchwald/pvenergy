from __future__ import annotations
from feature_accessor import FeatureAccessor
from pathlib import Path
import fsspec
import requests
from apidata import NSRDB_API_KEY, EMAIL # personal data not to be shared in repository
from io import StringIO
from tqdm import tqdm
from importlib_metadata import metadata
import s3fs

from feature_catalog import FeatureCatalog as F
from feature_catalog import Feature

import pandas as pd
from datetime import datetime
from typing import Any, Union

FeatureList = Union[Feature, tuple[Feature], list[Feature], None] 

class Pvdaq:
    fs = s3fs.S3FileSystem(anon=True)
    url = "oedi-data-lake/pvdaq/"
    SYSTEM_IDS = []

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
    DATA_COLUMNS = (F.PVDAQ_DC_POWER, F.PVDAQ_MODULE_TEMP, F.PVDAQ_POA_IRRADIANCE)

    @classmethod
    def load_metadata(cls) -> pd.DataFrame:
        """Load metadata like location, area, orientation
        for all IDs of pvdaq systems"""
        metadata_path = Path("metadata.csv")
        if metadata_path.exists():
            meta_df = pd.read_csv(metadata_path, index_col = F.SYSTEM_ID.name)
            cls.SYSTEM_IDS = meta_df.index.tolist()
            meta_df = meta_df.ftr.get()
            return meta_df

        prefix = cls.url + "parquet/"
        folders = ("mount", "site", "system")
        data = {}
        #load metadata files
        for name in folders:
            path = cls.fs.ls(prefix + name)[0]
            with cls.fs.open(path, "rb") as f:
                data[name] = pd.read_parquet("s3://" + path, filesystem = cls.fs)

        # merge metadata files
        metadata_df = (
            data["system"]
            .merge(data["site"], on="system_id", how="outer")
            .merge(data["mount"], on="system_id", how="outer")
        )
        metadata_df = metadata_df.rename(columns = cls.META_COLUMN_NAME_MAP)
        metadata_df = metadata_df.set_index(F.SYSTEM_ID.name)
        metadata_df = metadata_df.ftr.get()

        cls.SYSTEM_IDS = metadata_df.index.tolist()

        new_columns = metadata_df.columns.tolist() + [ftr.name for ftr in cls.DATA_COLUMN_PREFIX_MAP.values()]
        metadata_df = metadata_df.reindex(columns=new_columns, fill_value="")
        #data_col_dict = {"dc_power": F.PVDAQ_DC_POWER, "module_temp": F.PVDAQ_MODULE_TEMP, "poa_irradiance": F.PVDAQ_POA_IRRADIANCE}
        for id in tqdm(cls.SYSTEM_IDS, desc = "Loading PVDAQ metadata"):
            tqdm.write(f"PVDAQ system id {id}")
            df = cls.load_raw_data(id, file_limit=1, mute_tqdm = True)
            columns = df.columns.tolist()
            for prefix in cls.DATA_COLUMN_PREFIX_MAP.keys():
                metadata_df.at[id, cls.DATA_COLUMN_PREFIX_MAP[prefix].name] = cls._first_with(prefix, columns)         
        metadata_df.ftr.to_csv(metadata_path)
        return metadata_df

    @staticmethod
    def _first_with(prefix: str, string_list: list[str]):
        all_with = [s for s in string_list if s.lower().startswith(prefix)]
        if all_with:
            return all_with[0]
        return None

    @classmethod
    def meta(cls, system_id: int) -> dict[Feature, Any]:
        meta_df = cls.load_metadata()
        row = meta_df.loc[system_id]
        return {feature: row[feature.name] for feature in meta_df.ftr.features}

    @classmethod
    def filter_systems(cls, metacols: FeatureList = None) -> list[int]:
        """Filter pvdaq systems based on metadata criteria.
        Returns a list of system IDs that meet the criteria."""
        df = cls.load_metadata()
        df = df.ftr.dropna(metacols, how="any")
        return list(df.index)
    
    @classmethod
    def load_raw_data(cls, system_id: int,
                      file_limit: int | None = None,
                      use_columns: list | None = None,
                      rename_columns: list | None = None,
                      output_file: str | None = None,
                      mute_tqdm: bool = False) -> pd.DataFrame:
        """Load raw dataset of a pvdaq pv system with a given id for all recorded times.
        number of downloaded files and the set of used columns can be restricted."""
        prefix = cls.url + f"csv/pvdata/system_id={system_id}/"
        local_dir = Path("pvdata") / f"system_{system_id}"
        local_dir.mkdir(parents=True, exist_ok=True)

        files = cls.fs.glob(f"{prefix}**/*.csv")

        if not files:
            print(f"No data for pvdaq system {system_id} found")
            return pd.DataFrame()

        if file_limit is not None:
            files = files[:file_limit]

        dfs = []
        # Download all data files
        for remote_file in files if mute_tqdm else tqdm(files, desc=f"Loading PVDAQ csv files for system {system_id}"):
            if not mute_tqdm:
                tqdm.write(f"Loading file {remote_file}")
            local_file = local_dir / Path(remote_file).name
            if not local_file.exists():
                cls.fs.get(remote_file, local_file)
            try:
                df = pd.read_csv(local_file, parse_dates = ['measured_on'])
                if use_columns is not None:
                    df = df[[col for col in use_columns if col in df.columns]]
                if rename_columns is not None:
                    df = df.rename(columns = {col: name for (col, name) in zip(use_columns, rename_columns)})
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                print(f"Error while reading {local_file}: {e}")

        if not dfs:
            print(f"No valid CSV files found for pvdaq system {system_id}")
            return pd.DataFrame()

        full_df = pd.concat(dfs)

        # Save DataFrame as CSV
        if output_file is not None:
            full_df.to_csv(local_dir / output_file, index=False)

        return full_df
    
    @classmethod
    def load_measured_features(cls, system_id: int, file_limit: int | None = None, mute_tqdm = False) -> pd.DataFrame:
        """Load dataset of a pvdaq pv system with a given id containing DC power data for all recorded times."""
        meta = cls.meta(system_id)
        use_columns = ['measured_on'] + [meta[ftr] for ftr in cls.DATA_COLUMN_PREFIX_MAP.values() if not pd.isna(meta[ftr])]
        if len(use_columns) == 1:
            return pd.DataFrame()
        rename_columns = [F.TIME.name] + [ftr.name for ftr in cls.DATA_COLUMN_PREFIX_MAP.values() if not pd.isna(meta[ftr])]
        df = cls.load_raw_data(system_id, file_limit=file_limit, use_columns=use_columns, rename_columns=rename_columns, mute_tqdm = mute_tqdm)
        df[F.TIME.name] = pd.to_datetime(df[F.TIME.name])
        df = df.set_index(F.TIME.name)
        # Keep metadata except the original column names. They are irrelevant for further data analysis.
        df.ftr.set_const({ftr: val for (ftr, val) in meta.items() if ftr not in cls.DATA_COLUMNS})
        return df

class Nsrdb:
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
    

    @classmethod
    def load_year(cls, latitude: float, longitude: float,
                        year: int, overwrite_result: bool = False) -> pd.DataFrame:
        """Download NSRDB weather data for given GPS location and year.
        Saves data in weaterdata/ which gets loaded if the data is
        requested again (except when overwrite_result is True)."""

        output_root = Path("weatherdata")
        output_file = output_root / f"data_lat={latitude},lon={longitude},y={year}.csv"

        # Request data only if not already downloaded
        if not overwrite_result and output_file.exists():
            df = pd.read_csv(output_file, parse_dates = [F.TIME.name], index_col = F.TIME.name)
            return df.rename(columns = cls.COLUMN_NAME_MAP)

        output_metafile = output_root / f"meta_lat={latitude},lon={longitude},y={year}.csv"

        # Load all attributes that are relevant to calculate POA irridiance and temperature of a PV system
        attributes = ["air_temperature", "clearsky_dhi", "clearsky_dni", "clearsky_ghi",
                    "dhi", "dni", "ghi", "surface_albedo",                    
                    "wind_direction", "wind_speed"]
        #["cloud_fill_flag", "cloud_type", "dew_point","ozone", "relative_humidity","solar_zenith_angle", "ssa","surface_pressure", "total_precipitable_water"]

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
            raise ValueError("API request failed")

        # NSRDB metadata is so far unused
        meta = pd.read_csv(StringIO(response.text), nrows = 1)
        meta.to_csv(output_metafile)

        data = pd.read_csv(StringIO(response.text), skiprows = 2)#.dropna(axis = 1, how = "all")
        data.insert(0, F.TIME.name, pd.to_datetime(dict(
                    year=data['Year'],
                    month=data['Month'],
                    day=data['Day'],
                    hour=data['Hour'],
                    minute=data['Minute']
        )))
        data = data.drop(columns=['Year','Month','Day','Hour','Minute'])
        data = data.rename(columns = cls.COLUMN_NAME_MAP)
        data = data.set_index(F.TIME.name)
        data.to_csv(output_file)
        data.ftr.set_const({F.LATITUDE: latitude, F.LONGITUDE: longitude})
        
        return data

    @classmethod
    def load_time_range(cls, latitude: float, longitude: float,
                        start_date: pd.Timestamp, end_date: pd.Timestamp,
                        mute_tqdm: bool = False) -> pd.DataFrame:
        start_year = start_date.year
        end_year = end_date.year
        years = list(range(start_year, end_year+1))
        dfs = []
        for year in years if mute_tqdm else tqdm(years, desc=f"Loading weather data from {start_date}-{end_date} - CSVs"):
            if not mute_tqdm:
                tqdm.write(f"Loading weather data from year {year}")
            dfs.append(cls.load_year(latitude, longitude, year))
        
        dfs[0] = dfs[0][dfs[0].index >= start_date]
        dfs[-1] = dfs[-1][dfs[-1].index <= end_date]
        df = pd.concat(dfs)
        df.ftr.set_const({F.LATITUDE: latitude, F.LONGITUDE: longitude})

        return df

    @classmethod
    def load_system(cls, api: FeatureAccessor, mute_tqdm = False) -> pd.DataFrame:
        """Load weather data with calculated global poa for pv system with given system id."""
        start, end = api._df.index[0], api._df.index[-1]
        data = Nsrdb.load_time_range(api.get_const(F.LATITUDE), api.get_const(F.LONGITUDE), start, end, mute_tqdm = mute_tqdm)
        data.ftr.set_const(api.get_const())
        return data

def request_data(system_id: int, file_limit: int | None = None, mute_tqdm = False) -> pd.DataFrame:
    """Requests features from PVDAQ and NSRDB for system with given ID."""
    pv_data = Pvdaq.load_measured_features(system_id = system_id, file_limit = file_limit, mute_tqdm = mute_tqdm)
    if pv_data.empty:
        return pd.DataFrame()
    weather_data = Nsrdb.load_system(pv_data.ftr, mute_tqdm = mute_tqdm)
    df = pd.merge(pv_data, weather_data, left_index = True, right_index = True, how='inner')
    df.ftr.set_const(pv_data.ftr.get_const())
    return df

def get_features(system_id: int,
                features: FeatureList = None,
                file_limit: int | None = None,
                mute_tqdm = False) -> pd.DataFrame:
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
    print(request_data(2, file_limit=10))
