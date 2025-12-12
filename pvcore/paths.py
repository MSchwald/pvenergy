from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# The directories where the modules cache their data
DATA_DIR = BASE_DIR / "data"

# Requested data
RAW_DATA_DIR = DATA_DIR / "raw"
PVDAQ_DIR = RAW_DATA_DIR / "pvdaq"
NSRDB_DIR = RAW_DATA_DIR / "nsrdb"
OPENMETEO_DIR = RAW_DATA_DIR / "openmeteo"

# Data used for machine learning
MERGED_DIR = DATA_DIR / "merged"
RESULTS_DIR = DATA_DIR / "results"
TRAINING_DIR = DATA_DIR / "training"

# Trained ml models
MODELS_DIR = DATA_DIR / "models"

# Space for media for Django
MEDIA_DIR = BASE_DIR / "media"
STATIC_DIR = MEDIA_DIR / "static"

# Plots of weather, features, predictions
PLOTS_DIR = MEDIA_DIR / "plots"