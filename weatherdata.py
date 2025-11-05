import pandas as pd
import numpy as np
from datetime import datetime
import requests
from apidata import NSRDB_API_KEY, EMAIL # personal data not to be shared in repository
from pathlib import Path
from solardata import Pvdaq, PV_COL, META_COL
from io import StringIO
from tqdm import tqdm
import pvlib
from timezonefinder import TimezoneFinder
import pytz
import matplotlib.pyplot as plt

class WEATHER_COL:
    """Column names for weather data"""
    TIME = "time"
    # Parameters measured by Nsrdb
    AIR_TEMP = "air_temperature"
    DHI = "dhi"
    DNI = "dni"
    GHI = "ghi"
    SURFACE_ALBEDO = "surface_albedo"
    WIND_SPEED = "wind_speed"
    # Parameters calculated with pvlib
    SOLAR_ZENITH = "solar_zenith"
    SOLAR_AZIMUTH = "solar_azimuth"
    POA_IRRADIANCE = "poa_irradiance_calculated"
    AOI = "aoi_calculated"

class Nsrdb:
    tf = TimezoneFinder()
    
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
            return pd.read_csv(output_file, parse_dates = [WEATHER_COL.TIME])

        output_root.mkdir(exist_ok=True)
        output_metafile = output_root / f"meta_lat={latitude},lon={longitude},y={year}.csv"

        # Load all attributes that are relevant to calculate POA irridiance and temperature of a PV system
        attributes = [WEATHER_COL.AIR_TEMP, WEATHER_COL.DHI, WEATHER_COL.DNI, WEATHER_COL.GHI, WEATHER_COL.SURFACE_ALBEDO, WEATHER_COL.WIND_SPEED]

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

        meta = pd.read_csv(StringIO(response.text), nrows = 1)
        data = pd.read_csv(StringIO(response.text), skiprows = 2).dropna(axis = 1, how = "all")
        data.insert(0, WEATHER_COL.TIME, pd.to_datetime(dict(
                    year=data['Year'],
                    month=data['Month'],
                    day=data['Day'],
                    hour=data['Hour'],
                    minute=data['Minute']
        )))
        data = data.drop(columns=['Year','Month','Day','Hour','Minute'])
        data = data.rename(columns={"Temperature": WEATHER_COL.AIR_TEMP,
                             "Wind Speed": WEATHER_COL.WIND_SPEED,
                             "Surface Albedo": WEATHER_COL.SURFACE_ALBEDO,
                             "DHI": WEATHER_COL.DHI,
                             "DNI": WEATHER_COL.DNI,
                             "GHI": WEATHER_COL.GHI})
        
        meta.to_csv(output_metafile, index=False)
        data.to_csv(output_file, index=False)
        
        return data

    @classmethod
    def load_time_range(cls, latitude: float, longitude: float,
                        start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        start_year = start_date.year
        end_year = end_date.year
        years = list(range(start_year, end_year+1))
        dfs = []
        for year in tqdm(years, desc=f"Loading weather data from {start_date}-{end_date} - CSVs"):
            dfs.append(cls.load_year(latitude, longitude, year))
        dfs[0] = dfs[0][dfs[0][WEATHER_COL.TIME] >= start_date]
        dfs[-1] = dfs[-1][dfs[-1][WEATHER_COL.TIME] <= end_date]
        df = pd.concat(dfs, ignore_index=True)

        return df

    @classmethod
    def get_solarposition(cls, weather_df: pd.DataFrame,
                                pv_metadata: dict) -> pd.DataFrame:
        """For weather and meta data of a pv system, calculate solar position.
        Resulting DataFrame contains "azimuth" and "apparent_zenith" columns."""
        tz = pv_metadata[META_COL.TIME_ZONE]
        time = weather_df[WEATHER_COL.TIME].apply(lambda t: tz.localize(t, is_dst = False))
        solpos = pvlib.solarposition.get_solarposition(time = time,
                                                        latitude = pv_metadata[META_COL.LATITUDE],
                                                        longitude = pv_metadata[META_COL.LONGITUDE],
                                                        altitude = pv_metadata[META_COL.ELEVATION])
        solpos = solpos.rename(columns = {"apparent_zenith": WEATHER_COL.SOLAR_ZENITH,
                                 "azimuth": WEATHER_COL.SOLAR_AZIMUTH})
        return solpos

    @classmethod
    def calculate_energy_relevant_parameters(cls, weather_df: pd.DataFrame,
                        pv_metadata: dict) -> pd.DataFrame:
        """For weather and meta data of a pv system, calculate the apparent
        solar zenith angle, the plane of array irradiance
        and the module temperature on the considered pv module."""
        solpos = Nsrdb.get_solarposition(weather_df, pv_metadata)
        solpos.index = weather_df.index
        poa = pvlib.irradiance.get_total_irradiance(surface_tilt = pv_metadata[META_COL.TILT],
                                                surface_azimuth = pv_metadata[META_COL.AZIMUTH],
                                                solar_zenith = solpos[WEATHER_COL.SOLAR_ZENITH],
                                                solar_azimuth = solpos[WEATHER_COL.SOLAR_AZIMUTH],
                                                dni = weather_df[WEATHER_COL.DNI],
                                                ghi = weather_df[WEATHER_COL.GHI],
                                                dhi = weather_df[WEATHER_COL.DHI],
                                                albedo = weather_df[WEATHER_COL.SURFACE_ALBEDO])
        poa.index = weather_df.index
        aoi = pvlib.irradiance.aoi(surface_tilt = pv_metadata[META_COL.TILT],
                                surface_azimuth = pv_metadata[META_COL.AZIMUTH],
                                solar_zenith = solpos[WEATHER_COL.SOLAR_ZENITH],
                                solar_azimuth = solpos[WEATHER_COL.SOLAR_AZIMUTH])
        aoi.index = weather_df.index
        df = pd.concat([weather_df,
                        poa["poa_global"],
                        aoi],
                        axis=1)
        return df.rename(columns={"aoi": WEATHER_COL.AOI,
                           "poa_global": WEATHER_COL.POA_IRRADIANCE})

    @classmethod
    def load_system(cls, system_id: int,
                    start_date: pd.Timestamp,
                    end_date: pd.Timestamp) -> pd.DataFrame:
        """Load weather data with calculated global poa for pv system with given system id."""
        meta = Pvdaq.meta(system_id)
        weather_data = Nsrdb.load_time_range(meta[META_COL.LATITUDE], meta[META_COL.LONGITUDE], start_date, end_date)
        return Nsrdb.calculate_energy_relevant_parameters(weather_data, meta)

# Beispielaufruf:
if __name__ == "__main__":
    pv_data = Pvdaq.load_system(system_id=2, overwrite_result=True)
    start, end = pv_data[PV_COL.TIME].iloc[1], pv_data[PV_COL.TIME].iloc[-1]
    df = Nsrdb.load_system(2, start, end)
    print(pv_data.head(), df.head())
    df = pd.merge(pv_data, df, left_on=PV_COL.TIME, right_on=WEATHER_COL.TIME, how='inner').drop(columns=[WEATHER_COL.TIME])
    df.to_csv("test.csv", index=False)
    print(df.corr())
    
    """# Setze 'time' als Index für die Zeitreihen-Visualisierung
    df = df.set_index('measured_on')
    # ----------------------------------------------------------------------------------

    # Schaubild-Erstellung mit Matplotlib
    plt.figure(figsize=(12, 6))

    # Zeichne beide Zeitreihen
    plt.plot(df.index, df['poa_irradiance'], label='Measured irradiance (PVDAQ data)', color='blue')
    plt.plot(df.index, df['poa_global'], label='Modeled irradiance (NSRDB data)', color='red', linestyle='--')

    plt.title('Comparison of Plane of Array Irradiance (POA) over Time')
    plt.xlabel('Time')
    plt.ylabel('Irradiance ($\mathrm{W/m^2}$)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Speichere das Schaubild
    plot_filename = "poa_comparison_plot.png"
    plt.savefig(plot_filename)
    plt.close()
    """