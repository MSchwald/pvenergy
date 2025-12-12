from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from feature.accessor import Accessor as FeatureAccessor

import pandas as pd
import requests, os
from io import StringIO
from tqdm import tqdm

from pvcore.paths import NSRDB_DIR
from pvcore.feature import Catalog as F

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

        output_file = NSRDB_DIR / f"data_lat={latitude},lon={longitude},y={year}.csv"

        # Request data only if not already downloaded
        if output_file.exists():
            df = pd.read_csv(output_file, parse_dates = [F.TIME.name], index_col = F.TIME.name)
            return df.rename(columns = cls.COLUMN_NAME_MAP)

        #output_metafile = NSRDB_DIR / f"meta_lat={latitude},lon={longitude},y={year}.csv"

        # Load all attributes that are relevant to calculate POA irridiance and temperature of a PV system
        attributes = ["air_temperature", "clearsky_dhi", "clearsky_dni", "clearsky_ghi",
                    "dhi", "dni", "ghi", "surface_albedo",                    
                    "wind_direction", "wind_speed"]
        # Other available attributes, less interesting for us:
        #["cloud_fill_flag", "cloud_type", "dew_point","ozone", "relative_humidity","solar_zenith_angle",
        # "ssa","surface_pressure", "total_precipitable_water"]

        url = "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv"
        params = {
            "api_key": os.environ.get("NSRDB_API_KEY", "TRASH_API_KEY"),
            "wkt": f"POINT({longitude} {latitude})",
            "attributes": ",".join(attributes),
            "names": str(year),
            "utc": "false",
            "leap_day": "true",
            "email": os.environ.get("EMAIL", "TRASH_EMAIL"),
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
        cache_directory: str | None = NSRDB_DIR,
        cache_single_files: bool = False,
        mute_tqdm = False
    ) -> pd.DataFrame:
        """Load weather data with calculated global poa for pv system with given system id."""
        id = api.get_const(F.SYSTEM_ID)
        if cache_directory is not None:
            cache_directory.mkdir(parents = True, exist_ok = True)
            cache_path = NSRDB_DIR / f"weather_system_id={id}.parquet"
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