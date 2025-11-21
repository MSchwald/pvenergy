from __future__ import annotations
from pathlib import Path
from tqdm import tqdm
from importlib_metadata import metadata
import s3fs
import pyarrow.dataset as ds
import pandas as pd
import re
from natsort import natsorted

s3_fs = s3fs.S3FileSystem(anon=True)

def fs_from_path(path: str) -> object:
    """
    Utility function to make commands like fs.glob possible,
    where fs is either a local Path or an s3 fileserver.
    Could also be extended to http or others.
    """
    if Path(path).exists():
        return Path(path)
    if path.startswith("s3://"):
        return s3_fs
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
    if Path(file).exists():
        return read_csv_or_parquet(file)
    if cache is not None:
        local_path = Path(cache)
        if local_path.exists():
            return read_csv_or_parquet(cache)
        else:
            local_path.parent.mkdir(parents = True, exist_ok = True)
    df = read_csv_or_parquet(file)
    if cache is not None:
        if Path(cache).suffix == ".csv":
            df.to_csv(cache, index = False)
        elif Path(cache).suffix == ".parquet":
            df.to_parquet(cache, index = False)
    return df

def files_in(directory: str, file_format: str, sorted: bool = True) -> list:
    """lists all files in a local or remote directory (and subdirectories) of given format"""
    fs = fs_from_path(directory)
    if isinstance(fs, s3fs.S3FileSystem):
        files = list("s3://" + file for file in fs.glob(f"{directory}/**/*.{file_format}"))
    else:
        files = list(fs.glob(f"**/*.{file_format}"))
    if sorted:
        return natsorted(files, key = lambda f: str(f))
    return files

def concat_files(directory: str,
                file_format: str,
                cache_directory: str | None = None,
                cache_single_files: bool = False,
                use_columns: list | None = None,
                parquet_filter: list[tuple] | None = None,
                parquet_pivot: tuple[str, str, str, str] | None = None, #index, columns, values, aggfunc
                file_limit: int | None = None,
                mute_tqdm: bool = True) -> pd.DataFrame:
    """
    Load several csv or parquet files from a given local or remote directory,
    concat them and cache the result if a cache_directory is given.
    For loading parquet files, filter and pivot options can be specified.
    """
    if file_format not in ("parquet", "csv"):
        raise ValueError(f"Unknown format {file_format}.")    
    cache_file = None
    files = files_in(directory, file_format)
    N_files = len(files)
    if file_limit is not None:
        n_files = min(file_limit, N_files)
        files = files[:n_files]
    else:
        n_files = N_files
    base = Path(directory).name
    if cache_directory is not None:
        Path(cache_directory).mkdir(parents = True, exist_ok = True)
        if not cache_single_files:
            pattern = re.compile(rf"^{re.escape(base)}_{re.escape(file_format)}_(\d+)\.parquet$")
            candidates = []
            for file in Path(cache_directory).glob(f"{base}_{file_format}_*.parquet"):
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
    dfs = []
    if file_format == "csv":
        if cache_file is not None:
            dfs.append(cache_file)
        for file in files if mute_tqdm else tqdm(files, desc = f"Loading {file_format} files from {directory}"):
            if not mute_tqdm:
                tqdm.write(f"Loading file {file}")
            file_cache = str(Path(cache_directory) / (Path(file).stem + ".parquet")) if cache_single_files else None
            df = get_file(file, cache = file_cache)
            if use_columns is not None:
                df = df[[col for col in df.columns if col in use_columns]]
            if not df.empty:
                dfs.append(df)
        full_df = pd.concat(dfs, ignore_index = True)
    else:
        # For parquet format we use pyarrow for multithreading, which makes it much faster
        if not mute_tqdm:
            tqdm.write(f"Loading {len(files)} parquet files from {directory}")
        dataset = ds.dataset(
            files,
            format="parquet",
            filesystem = fs_from_path(directory)
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
            full_df = pd.concat([cache_file, full_df], ignore_index = True)
    if not cache_single_files and cache_directory is not None:
        full_df.to_parquet(Path(cache_directory) / f"{base}_{file_format}_{n_files}.parquet", index = False)
    return full_df