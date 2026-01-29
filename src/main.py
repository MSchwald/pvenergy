import sys, argparse, os
from pathlib import Path

from pvcore.io import Pvdaq
from pvcore.ml import Pipeline, ML_MODELS, Model
from pvcore.feature import Catalog as F, FEATURE_FROM_NAME, ALL_FEATURE_NAMES

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pvenergy.settings")

def main():
    default_features: list[str] = [
        ftr.name for ftr in [
            F.POWER_RATIO, F.PVLIB_POA_IRRADIANCE,
            F.ANNUAL_COSINUS, F.TIME_SINCE_SUNLIGHT,
            F.CLEAR_SKY_RATIO, F.COS_AOI, F.WIND_NORMAL_COMPONENT,
            F.POA_COS_AOI, F.DHI_PER_GHI,
            F.DCP_PER_AREA, F.GAMMA_TEMP_DIFFERENCE, F.RELATIVE_AZIMUTH
        ]
    ]
    default_models = [ML_MODELS.XGBOOST, ML_MODELS.LIGHTGBM, ML_MODELS.RANDOM_FOREST]
    model_names = [m.name for m in default_models]
    system_ids = Pvdaq.get_system_ids()

    parser = argparse.ArgumentParser(prog = "pvenergy", description = "PV Energy Forecasting")
    subparsers = parser.add_subparsers(dest = "command", required = True)
    
    django = subparsers.add_parser("django", help="Run arbitrary Django management commands", add_help=False)
    django.add_argument("django_args", nargs=argparse.REMAINDER)

    runserver = subparsers.add_parser("runserver", help="Shortcut for 'django runserver'", add_help=False)
    runserver.add_argument("django_args", nargs=argparse.REMAINDER)
    
    request = subparsers.add_parser("request", help="Requests raw data from PVDAQ and NSRDB")
    request.add_argument("--ids", nargs="+", type=int, default=Pipeline.TRAINING_IDS, choices=system_ids, help="PVDAQ system ids to request data for")

    train = subparsers.add_parser("train", help="Request data and train ML model")
    train.add_argument("--ids", nargs="+", type=int, default=Pipeline.TRAINING_IDS, choices=system_ids, help="PVDAQ system ids for training")
    train.add_argument("--features", nargs="+", default=default_features, choices=ALL_FEATURE_NAMES, help="Features to use for training")
    train.add_argument("--models", nargs="+", default=model_names, choices=model_names, help="ML models to train")
    train.add_argument("--tune", action="store_true", help="Use Optuna hyperparameter optimization")
    train.add_argument("--trials", type=int, default=10, help="Number of Optuna trials")
    train.add_argument("--cv", type=int, default=3, help="CV folds for hyperparameter tuning")

    evaluate = subparsers.add_parser("evaluate", help="System-wise feature importance analysis for trained model")
    evaluate.add_argument("--ids", nargs="+", type=int, default=Pipeline.TRAINING_IDS, choices=system_ids, help="PVDAQ system ids for evaluationr")
    evaluate.add_argument("--models", nargs="+", default=model_names, choices=model_names, help="ML models to evaluate")

    pipeline = subparsers.add_parser("pipeline", help="Do everything: Request, train, evaluate and open Django")
    pipeline.add_argument("--ids", nargs="+", type=int,default=Pipeline.TRAINING_IDS, choices=system_ids, help="PVDAQ system ids for training")
    pipeline.add_argument("--features", nargs="+", default=default_features, choices=ALL_FEATURE_NAMES, help="Features to use for training")
    pipeline.add_argument("--models", nargs="+", default=model_names, choices=model_names, help="ML models to train")
    pipeline.add_argument("--tune", action="store_true", help="Use Optuna hyperparameter optimization")
    pipeline.add_argument("--trials", type=int, default=10, help="Number of Optuna trials")
    pipeline.add_argument("--cv", type=int, default=3, help="CV folds for hyperparameter tuning")

    args = parser.parse_args()
    if args.command == "django":
        run_django(args.django_args)
    elif args.command == "runserver":
        django_args = args.django_args if args.django_args else ["0.0.0.0:8000"]
        run_django(["runserver", *django_args])
    elif args.command == "request":
        for system_id in args.ids:
            print(Pipeline.request_data(system_id))
    elif args.command == "train":
        for ml_model in [m for m in default_models if m.name in args.models]:
            Pipeline.fleet_analysis(
                system_ids = args.ids,
                training_features = tuple(FEATURE_FROM_NAME[name] for name in args.features),
                ml_model = ml_model,
                save_model_name = ml_model.name,
                tune = args.tune,
                n_trials = args.trials,
                cv = args.cv
            )
    elif args.command == "evaluate":
        for name in args.models:
            ml_model = Model.load(name)
            print(Pipeline.system_evaluations(trained_model = ml_model, system_ids = args.ids))
    elif args.command == "pipeline":
        for ml_model in [m for m in default_models if m.name in args.models]:
            Pipeline.fleet_analysis(
                system_ids = args.ids,
                training_features = tuple(FEATURE_FROM_NAME[name] for name in args.features),
                ml_model = ml_model,
                save_model_name = ml_model.name,
                tune = args.tune,
                n_trials = args.trials,
                cv = args.cv
            )
            Pipeline.system_evaluations(trained_model = ml_model, system_ids = args.ids)
        run_django(["runserver"])

def run_django(args):
    """Run django management commands"""
    from django.core.management import execute_from_command_line
    sys.argv = [os.path.abspath(__file__), *args]
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()
