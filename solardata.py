from pathlib import Path
import pandas as pd
import s3fs
from tqdm import tqdm
from timezonefinder import TimezoneFinder
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

class Pvdaq:
    fs = s3fs.S3FileSystem(anon=True)
    system_ids = []
    metacols = ['area', 'elevation', 'latitude', 'longitude', 'azimuth', 'tilt']
    tf = TimezoneFinder()

    @classmethod
    def load_metadata(cls, system_id: int | None = None) -> pd.DataFrame:
        """load metadata concerning location, area, orientation
        for all IDs of pvdaq systems"""
        metadata_path = Path("metadata.csv")
        if metadata_path.exists():
            df = pd.read_csv(metadata_path)
            if system_id is None:
                return df
            return(df[cls.metacols][df['system_id'] == system_id])

        prefix = "oedi-data-lake/pvdaq/parquet/"
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
            .merge(data["site"], on="system_id", how="left")
            .merge(data["mount"], on="system_id", how="left")
        )

        # save file as csv and retrun it as a dataframe
        metadata_df.to_csv(metadata_path, index = False)
        cls.system_ids = sorted(metadata_df["system_id"].tolist())
        cls.load_metadata(system_id)

    @classmethod
    def load_pvdata(cls, system_id: int, file_limit: int | None = None, overwrite_result: bool = False) -> pd.DataFrame:
        prefix = f"oedi-data-lake/pvdaq/csv/pvdata/system_id={system_id}/"
        output_root = Path("pvdata")
        output_root.mkdir(exist_ok=True)
        local_dir = output_root / f"system_{system_id}"
        local_dir.mkdir(parents=True, exist_ok=True)
        output_file = local_dir / "full.csv"

        if not overwrite_result and output_file.exists():
            if system_id == 2:
                return pd.read_csv(output_file, parse_dates = ['time'])
            return pd.read_csv(output_file, parse_dates = ['measured_on'])

        files = cls.fs.glob(f"{prefix}**/*.csv")

        if not files:
            print(f"Keine Dateien für System {system_id} gefunden")
            return pd.DataFrame()

        if file_limit is not None:
            files = files[:file_limit]

        if system_id == 2:
            read_cols = ["measured_on", "dc_power__346", "module_temp_1__349", "poa_irradiance__345"]
            save_cols = ["time", "dcp", "module_temp", "poa_irradiance"]
            df_full['time'] = pd.to_datetime(df_full['time'])
        else:
            df_full['measured_on'] = pd.to_datetime(df_full['measured_on'])

        # Alle Dateien herunterladen
        dfs = []
        for remote_file in tqdm(files, desc=f"Loading System {system_id} - CSVs"):
            local_file = local_dir / Path(remote_file).name
            if not local_file.exists():
                cls.fs.get(remote_file, local_file)
            try:
                df = pd.read_csv(local_file)
                if system_id == 2:
                    df = df[read_cols]
                    df.columns = save_cols
                dfs.append(df)
            except Exception as e:
                print(f"Fehler beim Lesen von {local_file}: {e}")

        if not dfs:
            print(f"Keine gültigen CSV-Dateien für System {system_id}")
            return pd.DataFrame()

        full_df = pd.concat(dfs, ignore_index=True)

        # DataFrame als CSV speichern
        full_df.to_csv(output_file, index=False)
        return full_df

    @classmethod
    def calculate_dcp0_gamma(cls):
        meta = Pvdaq.load_metadata(2)
        df_full = Pvdaq.load_pvdata(2)
        #["time", "dcp", "module_temp", "poa_irradiance"]
        df_full['dcp'] = df_full['dcp'].clip(lower=0)
        #df_full['poa_irradiance'] = df_full['poa_irradiance'].clip(lower=0)
        df_full['time'] = pd.to_datetime(df_full['time'])

        # Nur Tageswerte mit relevanter POA
        min_poa = 20
        df = df_full[df_full['poa_irradiance'] > min_poa].copy()

        df['day'] = df['time'].dt.date
        days = df['day'].unique()

        results = []

        for day in days:
            day_df = df[df['day'] == day]
            # nur gültige Zeilen behalten mit gemessener Sonnenstrahlung
            day_df = day_df.dropna(subset=['dcp', 'poa_irradiance', 'module_temp'])
            day_df = day_df[day_df['poa_irradiance'] > 100]

            # prüfen, ob noch genügend Datenpunkte für Regression übrig sind
            if len(day_df) < 5:
                continue  # überspringen

            # Input-Features für lineare Regression
            # dcp = pdc0 * (1 + gamma * (T_cell - 25)) * POA/1000
            X = (day_df['module_temp'] - 25).values.reshape(-1,1)  # delta T
            poa = (day_df['poa_irradiance'] / 1000).values.reshape(-1,1)
            X_fit = np.hstack([poa, poa*X])
            y = day_df['dcp'].values
            
            model = LinearRegression(fit_intercept=False)
            model.fit(X_fit, y)
            
            pdc0 = model.coef_[0]
            if pdc0 != 0:
                gamma = model.coef_[1] / pdc0
            else:
                pdc0 = None
                gamma = None

            # Realistische Werte begrenzen
            if pdc0 is not None and (pdc0 <= 0 or pdc0 > meta['area'].iloc[0]*100):  # grob: max 100 W/m² pro m²
                pdc0 = None
                gamma = None

            results.append({"date": day, "pdc0": pdc0, "gamma": gamma})

        # Als CSV abspeichern
        results_df = pd.DataFrame(results)
        results_df.to_csv("pdc0_gamma_per_day.csv", index=False)

# Beispielaufruf:
if __name__ == "__main__":
    meta = Pvdaq.load_metadata(2)
    df_full = Pvdaq.load_pvdata(2)
    df_full['dcp'] = df_full['dcp'].clip(lower=0)
    df_full['poa_irradiance'] = df_full['poa_irradiance'].clip(lower=0)
    df_full['time'] = pd.to_datetime(df_full['time'])
    df_full['day'] = df_full['time'].dt.date

    gleich_laufende_tage = []

    for day, day_df in df_full.groupby('day'):
        poa_thresh = 50  # W/m², definieren wann Sonne “einsetzt”
        dcp_thresh = 1   # kW oder was sinnvoll ist

        poa_start = day_df[day_df['poa_irradiance'] > poa_thresh]['time'].min()
        dcp_start = day_df[day_df['dcp'] > dcp_thresh]['time'].min()

    if pd.notna(poa_start) and pd.notna(dcp_start):
        delta = (dcp_start - poa_start).total_seconds() / 60  # Minuten Unterschied
        if delta <= 15:  # Schwellwert
            gleich_laufende_tage.append(day)

    print(gleich_laufende_tage)