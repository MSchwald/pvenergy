from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from pvcore.paths import TRAINING_DIR
from pvcore.io import request_data, Pvdaq, Nsrdb, OpenMeteo
from pvcore.feature import Source, Feature, Catalog as F, Processing
from pvcore.ml import Pipeline, EvaluationMethod, EVALUATIONS, ALL_EVALUATIONS, Scaler, Model, ML_MODELS
from pvcore.plotting import Plot

if __name__ == '__main__':
    # Current Training features for the ML models
    training_features: list[Feature] = [
        F.POWER_RATIO, F.PVLIB_POA_IRRADIANCE,
        F.DAY_OF_YEAR, F.TIME_SINCE_SUNLIGHT,
        F.CLEAR_SKY_RATIO, F.COS_AOI, F.WIND_NORMAL_COMPONENT,
        F.POA_COS_AOI, F.POA_WIND_SPEED, F.DHI_PER_GHI,
        F.DCP_PER_AREA, F.GAMMA_TEMP_DIFFERENCE, F.RELATIVE_AZIMUTH
    ]
    
    for ml_model in [ML_MODELS.XGBOOST, ML_MODELS.LIGHTGBM, ML_MODELS.RANDOM_FOREST]:
        Pipeline.fleet_analysis(
                system_ids = Pipeline.TRAINING_IDS,
                training_features = training_features,
                target_feature = F.PVDAQ_DC_POWER,
                clip_features = {F.PVDAQ_DC_POWER: (0, None)},
                filter_features = {F.PVDAQ_DC_POWER: (0, 3000), F.PVLIB_POA_IRRADIANCE: (1, None)},
                ml_model = ml_model,
                file_limit = None,
                mute_tqdm = False,
                training_data_cache =  TRAINING_DIR / "full_training_data.parquet",
                hyper_parameter_search = False,
                use_cached_training_data = True,
                save_model_name = ml_model.name
        )

        """res = Pipeline.individual_analysis(
            system_ids = Pipeline.TRAINING_IDS,
            training_features = training_features,
            target_feature = F.PVDAQ_DC_POWER,
            clip_features = {F.PVDAQ_DC_POWER: (0, None)},
            filter_features = {F.PVDAQ_DC_POWER: (0, 3000), F.PVLIB_POA_IRRADIANCE: (1, None)},
            ml_model = ml_model,
            file_limit = None,
            mute_tqdm = False,
            training_data_cache_dir = TRAINING_DIR,
            hyper_parameter_search = False,
            use_cached_training_data = False,
            save_model_name = None
        )
        
        print(res)"""