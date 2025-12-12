from __future__ import annotations

from pathlib import Path
from tqdm import tqdm
import s3fs
import pyarrow.dataset as ds
import pandas as pd
import re
from natsort import natsorted
from datetime import datetime, timedelta

from pvcore.paths import BASE_DIR

s3_fs = s3fs.S3FileSystem(anon=True)

def absolute_path(path: str | Path) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()

def fs_from_path(path: str) -> object:
    """
    Utility function to make commands like fs.glob possible,
    where fs is either a local Path or an s3 fileserver.
    Could also be extended to http or others.
    """
    if path.startswith("s3://"):
        return s3_fs
    p = absolute_path(path)
    if p.exists():
        return p
    raise ValueError(f"Unknown path type: {path}")

def read_csv_or_parquet(file: str) -> pd.DataFrame:
    """
    Loads csv or parquet files from local or remote path.
    """
    path = Path(file)
    file_format = path.suffix
    if not file_format in (".csv", ".parquet"):
        raise ValueError(f"Unknown format {file_format}.")
    if file_format == ".csv":
        return pd.read_csv(file)
    elif file_format == ".parquet":
        return pd.read_parquet(file)

def get_file(file: str, cache: str | None = None) -> pd.DataFrame:
    """
    Loads csv or parquet files from path or url.
    recognizes local files and links to Pvdaq.
    If a cache path is provided, remote files are
    only requested if not already cached.
    """
    path = absolute_path(file)
    if path.exists():
        return read_csv_or_parquet(path)
    if cache is not None:
        local_path = absolute_path(cache)
        if local_path.exists():
            return read_csv_or_parquet(cache)
        else:
            local_path.parent.mkdir(parents = True, exist_ok = True)
    df = read_csv_or_parquet(file)
    if cache is not None:
        if local_path.suffix == ".csv":
            df.to_csv(cache, index = False)
        elif local_path.suffix == ".parquet":
            df.to_parquet(cache, index = False)
    return df

def files_in(directory: str, file_format: str, sorted: bool = True) -> list[str]:
    """lists all files in a local or remote directory (and subdirectories) of given format"""
    fs = fs_from_path(directory)
    if isinstance(fs, s3fs.S3FileSystem):
        files = list("s3://" + file for file in fs.glob(f"{directory}/**/*.{file_format}"))
    else:
        files = list(fs.glob(f"**/*.{file_format}"))
    if sorted:
        return natsorted(files, key = lambda f: str(f))
    return files

def concat_filelist(
    files: list[str],
    file_format: str,
    cache_file: str | None = None,
    single_file_cache_dir: str | None = None,
    use_columns: list | None = None,
    parquet_filter: list[tuple] | None = None,
    parquet_pivot: tuple[str, str, str, str] | None = None, #index, columns, values, aggfunc
    mute_tqdm: bool = False
) -> pd.DataFrame:
    """
    Load several csv or parquet files from a given local or remote directory,
    concat them and cache the result if a cache_directory is given.
    For loading parquet files, filter and pivot options can be specified.
    """
    if file_format not in ("parquet", "csv"):
        raise ValueError(f"Unknown format {file_format}.")
    if cache_file is not None:
        cache_file = absolute_path(cache_file)
        if cache_file.exists():
            return get_file(cache_file)
        cache_file.parent.mkdir(parents = True, exist_ok = True)
    if single_file_cache_dir is not None:
        single_file_cache_dir = absolute_path(single_file_cache_dir)
    dfs = []
    if file_format == "csv" or single_file_cache_dir is not None:
        for file in files if mute_tqdm else tqdm(files, desc = f"Loading {len(files)} {file_format} files"):
            if not mute_tqdm:
                tqdm.write(f"Loading file {file}")
            file_cache = single_file_cache_dir / f"{Path(file).stem}.parquet" if single_file_cache_dir is not None else None
            df = get_file(file, cache = file_cache)
            if use_columns is not None:
                df = df[[col for col in df.columns if col in use_columns]]
            if not df.empty:
                dfs.append(df)
        full_df = pd.concat(dfs, ignore_index = True)
    else:
        # For parquet format we use pyarrow for multithreading, which makes it much faster
        if not mute_tqdm:
            tqdm.write(f"Loading {len(files)} parquet files")
        fs = fs_from_path(files[0])
        dataset = ds.dataset(
            files,
            format = "parquet",
            filesystem = fs
        )
        table = dataset.to_table(
            columns = use_columns,
            filter = parquet_filter,
        )
        full_df = table.to_pandas()
        if parquet_pivot is not None:
            index, columns, values, aggfunc = parquet_pivot
            full_df = full_df.pivot_table(
                index = index,
                columns = columns,
                values = values,
                aggfunc = aggfunc
            ).reset_index()
            full_df.columns.name = None
            full_df = full_df.rename(columns = {col: str(col) for col in full_df.columns})
    if cache_file is not None:
        full_df.to_parquet(cache_file, index = False)
    return full_df

def concat_files(
    directory: str,
    file_format: str,
    cache_directory: str | None = None,
    cache_single_files: bool = False,
    use_columns: list | None = None,
    parquet_filter: list[tuple] | None = None,
    parquet_pivot: tuple[str, str, str, str] | None = None, #index, columns, values, aggfunc
    file_limit: int | None = None,
    mute_tqdm: bool = False
) -> pd.DataFrame:
    """
    Load several csv or parquet files from a given local or remote directory,
    concat them and cache the result if a cache_directory is given.
    For loading parquet files, filter and pivot options can be specified.
    """
    path = None
    base = f"{Path(directory).name}_{file_format}_"
    if cache_directory is not None:
        path = absolute_path(cache_directory)
        path.mkdir(parents = True, exist_ok = True)
    if file_limit is None:
        if cache_directory is not None:
            if (path / f"{base}full.parquet").exists():
                return pd.read_parquet(path / f"{base}full.parquet")
    files = files_in(directory, file_format)
    N_files = len(files)
    if file_limit is not None:
        n_files = min(file_limit, N_files)
        files = files[:n_files]
    else:
        n_files = N_files
    if cache_single_files:
        return concat_filelist(
            files = files, file_format = file_format, use_columns = use_columns, single_file_cache_dir = path,
            parquet_filter = parquet_filter, parquet_pivot = parquet_pivot, mute_tqdm = mute_tqdm
        )
    if path is not None:  
        output_file = path / f"{base}full.parquet" if n_files == N_files else path / f"{base}{n_files}.parquet"
        if output_file.exists():
            return pd.read_parquet(output_file)
    else:
        output_file = None
    cache_file = None
    if path is not None:
        pattern = re.compile(rf"^{re.escape(base)}(\d+)\.parquet$")
        candidates = []
        for file in path.glob(f"{base}*.parquet"):
            match = pattern.match(file.name)
            if match:
                n = int(match.group(1))
                if n <= n_files:
                    candidates.append((n, file))
        if candidates:
            n_caches, cache_file_path = max(candidates, key = lambda pair: pair[0])
            files = files[n_caches:]
            cache_file = pd.read_parquet(cache_file_path)
            if not files:
                return cache_file
    
    full_df = concat_filelist(
        files = files, file_format = file_format, use_columns = use_columns,
        parquet_filter = parquet_filter, parquet_pivot = parquet_pivot, mute_tqdm = mute_tqdm
    )
    if cache_file is not None:
        full_df = pd.concat([cache_file, full_df], ignore_index = True)
    
    if not cache_single_files and output_file is not None:
        full_df.to_parquet(output_file, index = False)
    return full_df

def file_up_to_date(file: str | Path):
    path = absolute_path(file)
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(file.stat().st_mtime)
    time_now = datetime.now()
    if time_now - mtime < timedelta(hours = 1) and time_now.hour == mtime.hour:
        return True
    return False