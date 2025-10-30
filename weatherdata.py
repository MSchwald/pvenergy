from datetime import datetime
import requests
from apidata import NSRDB_API_KEY, EMAIL # personal data not to be shared in repository
import pandas as pd
from pathlib import Path
from solardata import Pvdaq
from io import StringIO
from tqdm import tqdm

class Nsrdb:

    @classmethod
    def load_year(cls, latitude: float, longitude: float,
                        year: int, overwrite_result: bool = False) -> pd.DataFrame:
        """Download NSRDB weather data for given GPS location and year."""

        output_root = Path("weatherdata")
        output_file = output_root / f"data_lat={latitude},lon={longitude},y={year}.csv"

        # request data only if not already loaded
        if not overwrite_result and output_file.exists():
            data = pd.read_csv(output_file, parse_dates = ['time'])

            return data

        output_root.mkdir(exist_ok=True)
        output_metafile = output_root / f"meta_lat={latitude},lon={longitude},y={year}.csv"

        attributes = "air_temperature,dhi,dni,ghi,surface_albedo,wind_speed"

        url = (
            f"https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv?"
            f"api_key={NSRDB_API_KEY}&"
            f"wkt=POINT({longitude} {latitude})&"
            f"attributes={attributes}&"
            f"names={year}&"
            f"utc=false&"
            f"leap_day=true&"
            f"email={EMAIL}"
        )

        response = requests.get(url)
        response.raise_for_status()

        meta = pd.read_csv(StringIO(response.text), nrows = 1)
        data = pd.read_csv(StringIO(response.text), skiprows = 2).dropna(axis = 1, how = "all")
        data.insert(0, 'time', pd.to_datetime(dict(
                    year=data['Year'],
                    month=data['Month'],
                    day=data['Day'],
                    hour=data['Hour'],
                    minute=data['Minute']
        )))
        data = data.drop(columns=['Year','Month','Day','Hour','Minute'])

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
        dfs[0] = dfs[0][dfs[0]['time'] >= start_date]
        dfs[-1] = dfs[-1][dfs[-1]['time'] <= end_date]
        df = pd.concat(dfs, ignore_index=True)

        return df


# Beispielaufruf:

if __name__ == "__main__":
    meta = Pvdaq.load_metadata(2)
    latitude, longitude = meta['latitude'].squeeze(), meta['longitude'].squeeze()
    print(latitude, longitude)
    #df = Pvdaq.load_pvdata(2)
    #start, end = df['time'].iat[0], df['time'].iat[-1]
    #data = Nsrdb.load_year(latitude, longitude, 2012)
    #data = Nsrdb.load_time_range(latitude, longitude, start, end)
    
    #print(data, data.columns, data.shape)





