from pathlib import Path
import pandas as pd
from pathlib import Path

from pvcore.feature import Processing as fp
from typing import Any

from django.conf import settings

def number_format(number: float | int | Any) -> str | Any:
    if isinstance(number, pd.Timestamp):
        return number.strftime('%X')
    if not isinstance(number, (float, int)):
        return number
    s = str(number)
    if "." not in s:
        return s
    _, mantissa = s.split(".")
    if len(mantissa) <= 2:
        return s
    return f"{round(number, 2):.2f}"

def feature_format(name: str, display_unit: bool = True) -> str:
    if not isinstance(name, str):
        return name
    if not name in fp.ALL_FEATURE_NAMES:
        return name.replace("_", " ").title()
    if display_unit:
        return fp.FEATURE_FROM_NAME[name].display_name_with_unit
    return fp.FEATURE_FROM_NAME[name].display_name

def pd_styler(data: pd.DataFrame | pd.Series) -> str:
    """Formats pandas objects with html code for displaying."""
    df = data.copy()
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    df.columns = [feature_format(col) for col in df.columns]
    df.columns.name, df.index.name = feature_format(df.index.name), None
    df_html = df.style.format(
        formatter=number_format
    ).format_index(
        number_format
    ).to_html(
        escape=False, table_attributes='class="df-table"'
    )
    return f'<div class="table-container">{df_html}</div>'

def file_to_url(file: Path) -> str:
    """Converts a local path into a public url."""
    try:
        rel = file.relative_to(settings.MEDIA_ROOT)
        print(f"{settings.MEDIA_URL}{rel.as_posix()}")
        return f"{settings.MEDIA_URL}{rel.as_posix()}"
    except ValueError:
        pass
    try:
        rel = file.relative_to(settings.STATIC_ROOT)
        return f"{settings.STATIC_URL}{rel.as_posix()}"
    except (ValueError, AttributeError):
        pass
    print(f"Warning: file {file} neither in media nor staticfiles folder.")
    