from pathlib import Path
from importlib_metadata import metadata
import pandas as pd
import s3fs
from tqdm import tqdm
from timezonefinder import TimezoneFinder
import pytz
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

class PV_COL:
    """Column names for the measured pv data of pvdaq pv systems"""
    TIME = "measured_on"
    DC_POWER = "dc_power_measured"
    MODULE_TEMP = "module_temp_measured"
    POA_IRRADIANCE = "poa_irradiance_measured"

class META_COL:
    """Column names for the metadata of pvdaq pv systems"""
    SYSTEM_ID = "system_id"
    AREA = "area"
    ELEVATION = "elevation"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    AZIMUTH = "azimuth"
    TILT = "tilt"
    TIME_ZONE = "time_zone"
    DCP_COLUMN = "dcp_column"
    MT_COLUMN = "mt_column"



class Pvdaq:
    fs = s3fs.S3FileSystem(anon=True)
    url = "oedi-data-lake/pvdaq/"
    metacols = [META_COL.AREA, META_COL.ELEVATION, META_COL.LATITUDE, META_COL.LONGITUDE, META_COL.AZIMUTH, META_COL.TILT]
    tf = TimezoneFinder()
    SYSTEM_IDS = []

    @classmethod
    def load_metadata(cls) -> pd.DataFrame:
        """load metadata concerning location, area, orientation
        for all IDs of pvdaq systems"""
        metadata_path = Path("metadata.csv")
        if metadata_path.exists():
            meta_df = pd.read_csv(metadata_path, usecols = cls.metacols + [META_COL.SYSTEM_ID], index_col = META_COL.SYSTEM_ID)
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
            .merge(data["site"], on=META_COL.SYSTEM_ID, how="outer")
            .merge(data["mount"], on=META_COL.SYSTEM_ID, how="outer")
        )
        metadata_df = metadata_df.set_index(META_COL.SYSTEM_ID)
        cls.SYSTEM_IDS = metadata_df.index.tolist()

        new_columns = metadata_df.columns.tolist() + [META_COL.DCP_COLUMN, META_COL.MT_COLUMN]
        metadata_df = metadata_df.reindex(columns=new_columns, fill_value=None)
        for id in cls.SYSTEM_IDS:
            df = cls.load_raw_data(id, file_limit=1)
            columns = df.columns.tolist()
            dcp_columns = [col for col in columns if col.lower().startswith("dc_power")]
            if dcp_columns:
                metadata_df.loc[id, META_COL.DCP_COLUMN] = dcp_columns[0]
            mt_columns = [col for col in columns if col.lower().startswith("module_temp")]
            if mt_columns:
                metadata_df.loc[id, META_COL.MT_COLUMN] = mt_columns[0]
         
        # save file as csv and retrun it as a dataframe
        metadata_df.to_csv(metadata_path, index = False)
        return metadata_df

    @classmethod
    def meta(cls, system_id: int) -> dict:
        meta_df = cls.load_metadata()
        return meta_df.loc[system_id].to_dict()

    @classmethod
    def filter_sytems(cls, metacols: list | None = None) -> list[int]:
        """Filter pvdaq systems based on metadata criteria.
        Returns a list of system IDs that meet the criteria."""
        if metacols is None:
            metacols = cls.metacols

        df = cls.load_metadata()[metacols]
        df = df.dropna(how="any")
        return list(df.index)
    
    @classmethod
    def load_raw_data(cls, system_id: int,
                      file_limit: int | None = None,
                      use_columns: list | None = None, rename_columns: list | None = None,
                      output_file: str | None = None) -> pd.DataFrame:
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
        for remote_file in tqdm(files, desc=f"Loading System {system_id} - CSVs"):
            local_file = local_dir / Path(remote_file).name
            if not local_file.exists():
                cls.fs.get(remote_file, local_file)
            try:
                df = pd.read_csv(local_file, parse_dates = ['measured_on'])
                if use_columns is not None:
                    df = df[use_columns]
                if rename_columns is not None:
                    df.columns = rename_columns
                dfs.append(df)
            except Exception as e:
                print(f"Error while reading {local_file}: {e}")

        if not dfs:
            print(f"No valid CSV files found for pvdaq system {system_id}")
            return pd.DataFrame()

        full_df = pd.concat(dfs, ignore_index=True)

        # Save DataFrame as CSV
        if output_file is not None:
            full_df.to_csv(local_dir / output_file, index=False)

        return full_df
    
    @classmethod
    def load_dcp_module_temp_data(cls, system_id: int, file_limit: int | None = None) -> pd.DataFrame:
        """Load dataset of a pvdaq pv system with a given id containing DC power data for all recorded times."""
        dcp_column = cls.meta(system_id)[META_COL.DCP_COLUMN]
        mt_column = cls.meta(system_id)[META_COL.MT_COLUMN]
        if dcp_column is None:
            print(f"No DC power column found for system {system_id}")
            return pd.DataFrame()

        use_columns = ['measured_on', dcp_column]
        rename_columns = [PV_COL.TIME, PV_COL.DC_POWER]
        if mt_column is not None:
            use_columns.append(mt_column)
            rename_columns.append(PV_COL.MODULE_TEMP)
        return cls.load_raw_data(system_id, file_limit=file_limit, use_columns=use_columns, rename_columns=rename_columns)

    @classmethod
    def load_system2(cls, file_limit: int | None = None, output_file: str | None = None) -> pd.DataFrame:
        """Load dataset of a nsrdb pv system with id 2 from pvdaq for all recorded times."""
        return cls.load_raw_data(2, file_limit=file_limit,
                          use_columns=["measured_on", "dc_power__346", "module_temp_1__349", "poa_irradiance__345"],
                          rename_columns=[PV_COL.TIME, PV_COL.DC_POWER, PV_COL.MODULE_TEMP ,"poa_irradiance_measured"],
                          output_file=output_file)

# Tests
if __name__ == "__main__":
    meta = Pvdaq.meta(3)
    #print(Pvdaq.load_dcp_data(3, file_limit=5).head())
    #latitude, longitude = meta['latitude'], meta['longitude']

"""    df_full = Pvdaq.load_system(2, overwrite_result=True)
    df_full['dcp'] = df_full['dcp'].clip(lower=0)
    df_full['poa_irradiance_measured'] = df_full['poa_irradiance_measured'].clip(lower=0)
    df_full['measured_on'] = pd.to_datetime(df_full['measured_on'])
    df_full['day'] = df_full['measured_on'].dt.date

    gleich_laufende_tage = []

    for day, day_df in df_full.groupby('day'):
        poa_thresh = 50  # W/m², definieren wann Sonne “einsetzt”
        dcp_thresh = 1   # kW oder was sinnvoll ist

        poa_start = day_df[day_df['poa_irradiance_measured'] > poa_thresh]['measured_on'].min()
        dcp_start = day_df[day_df['dcp_measured'] > dcp_thresh]['measured_on'].min()

    if pd.notna(poa_start) and pd.notna(dcp_start):
        delta = (dcp_start - poa_start).total_seconds() / 60  # Minuten Unterschied
        if delta <= 15:  # Schwellwert
            gleich_laufende_tage.append(day)

    print(gleich_laufende_tage)"""