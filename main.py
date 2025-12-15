import sys, argparse
from pathlib import Path
from dotenv import load_dotenv

from pvcore.io import Pvdaq
from pvcore.ml import Pipeline, ML_MODELS, Model
from pvcore.feature import Feature, Catalog as F, FEATURE_FROM_NAME, ALL_FEATURE_NAMES

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

default_features: list[Feature] = [
    ftr.name for ftr in [
        F.POWER_RATIO, F.PVLIB_POA_IRRADIANCE,
        F.DAY_OF_YEAR, F.TIME_SINCE_SUNLIGHT,
        F.CLEAR_SKY_RATIO, F.COS_AOI, F.WIND_NORMAL_COMPONENT,
        F.POA_COS_AOI, F.POA_WIND_SPEED, F.DHI_PER_GHI,
        F.DCP_PER_AREA, F.GAMMA_TEMP_DIFFERENCE, F.RELATIVE_AZIMUTH
    ]
]
default_models = [ML_MODELS.XGBOOST, ML_MODELS.LIGHTGBM, ML_MODELS.RANDOM_FOREST]
model_names = [m.name for m in default_models]
system_ids = [str(id) for id in Pvdaq.get_system_ids()]

def main():
    parser = argparse.ArgumentParser(prog = "main", description = "PV Energy Forecasting")
    subparsers = parser.add_subparsers(dest = "command", required = True)
    
    django = subparsers.add_parser("django", help="Run arbitrary Django management commands", add_help=False)
    django.add_argument("django_args", nargs=argparse.REMAINDER)

    runserver = subparsers.add_parser("runserver", help="Shortcut for 'django runserver'", add_help=False)
    runserver.add_argument("django_args", nargs=argparse.REMAINDER)
    
    request = subparsers.add_parser("request", help="Requests raw data from PVDAQ and NSRDB")
    request.add_argument("--ids", nargs="+", default=Pipeline.TRAINING_IDS, choices=system_ids, help="PVDAQ system ids to request data for")

    train = subparsers.add_parser("train", help="Request data and train ML model")
    train.add_argument("--ids", nargs="+", default=Pipeline.TRAINING_IDS, choices=system_ids, help="PVDAQ system ids for training")
    train.add_argument("--features", nargs="+", default=default_features, choices=ALL_FEATURE_NAMES, help="Features to use for training")
    train.add_argument("--models", nargs="+", default=model_names, choices=model_names, help="ML models to train")

    evaluate = subparsers.add_parser("evaluate", help="System-wise feature importance analysis for trained model")
    evaluate.add_argument("--ids", nargs="+", default=Pipeline.TRAINING_IDS, choices=system_ids, help="PVDAQ system ids for evaluationr")
    evaluate.add_argument("--models", nargs="+", default=model_names, choices=model_names, help="ML models to evaluate")

    pipeline = subparsers.add_parser("pipeline", help="Do everything: Request, train, evaluate and open Django")
    pipeline.add_argument("--ids", nargs="+", default=Pipeline.TRAINING_IDS, choices=system_ids, help="PVDAQ system ids for training")
    pipeline.add_argument("--features", nargs="+", default=default_features, choices=ALL_FEATURE_NAMES, help="Features to use for training")
    pipeline.add_argument("--models", nargs="+", default=model_names, choices=model_names, help="ML models to train")

    args = parser.parse_args()
    if args.command == "django":
        run_django(args.django_args)
    elif args.command == "runserver":
        run_django(["runserver", *args.django_args])
    elif args.command == "request":
        for system_id in args.ids:
            print(Pipeline.request_data(system_id))
    elif args.command == "train":
        for ml_model in [m for m in default_models if m.name in args.models]:
            Pipeline.fleet_analysis(
                system_ids = args.ids,
                training_features = [FEATURE_FROM_NAME[name] for name in args.features],
                ml_model = ml_model,
                save_model_name = ml_model.name
            )
    elif args.command == "evaluate":
        for name in args.models:
            ml_model = Model.load(name)
            Pipeline.system_evaluations(trained_model = ml_model, system_ids = args.ids)
    elif args.command == "pipeline":
        for ml_model in [m for m in default_models if m.name in args.models]:
            Pipeline.fleet_analysis(
                system_ids = args.ids,
                training_features = [FEATURE_FROM_NAME[name] for name in args.features],
                ml_model = ml_model,
                save_model_name = ml_model.name
            )
            Pipeline.system_evaluations(trained_model = ml_model, system_ids = args.ids)
        run_django(["runserver"])

def run_django(args):
    """Run django management commands"""
    from django.core.management import execute_from_command_line
    sys.argv = [Path(sys.argv[0]).name, *args]
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()
