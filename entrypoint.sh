#!/bin/bash
set -e

MODELS_DIR="/pvenergy/data/models"
MISSING_MODELS=()

[ ! -f "$MODELS_DIR/xgboost.joblib" ] && MISSING_MODELS+=("xgboost")
[ ! -f "$MODELS_DIR/lightgbm.joblib" ] && MISSING_MODELS+=("lightgbm")
[ ! -f "$MODELS_DIR/random_forest.joblib" ] && MISSING_MODELS+=("random_forest")

if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
    pvenergy train --models "${MISSING_MODELS[@]}"
fi

exec "$@"