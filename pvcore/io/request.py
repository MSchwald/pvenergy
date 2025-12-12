from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from feature.accessor import Accessor as FeatureAccessor, FeatureList

import pandas as pd

from pvcore.paths import MERGED_DIR
from .pvdaq import Pvdaq
from .nsrdb import Nsrdb
from . import file_utilities as fu

def request_data(
    system_id: int,
    file_limit: int | None = None,
    output_dir: str | None = MERGED_DIR,
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