from . import feature, paths
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

for attr_name in dir(paths):
    attr = getattr(paths, attr_name)
    if isinstance(attr, Path) and not attr.suffix:
        attr.mkdir(parents = True, exist_ok = True)
