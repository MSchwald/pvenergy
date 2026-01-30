#!/bin/bash
set -e

DATA_DIR="/pvenergy/data"
DEFAULTS_DIR="/pvenergy/defaults"
MODELS_DIR="$DATA_DIR/models"
TRAINING_DIR="$DATA_DIR/training"
RESULTS_DIR="$DATA_DIR/results"

sync_defaults() {
    local target_dir=$1
    local source_dir=$2
    if [ -d "$target_dir" ] && [ -z "$(ls -A "$target_dir")" ]; then
        cp -r "$source_dir"/. "$target_dir"/
    fi
}

sync_defaults "$TRAINING_DIR" "$DEFAULTS_DIR/training"
sync_defaults "$RESULTS_DIR" "$DEFAULTS_DIR/results"

MISSING_MODELS=()
[ ! -f "$MODELS_DIR/xgboost.joblib" ] && MISSING_MODELS+=("xgboost")
[ ! -f "$MODELS_DIR/lightgbm.joblib" ] && MISSING_MODELS+=("lightgbm")
[ ! -f "$MODELS_DIR/random_forest.joblib" ] && MISSING_MODELS+=("random_forest")

if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
    pvenergy train --models "${MISSING_MODELS[@]}"
fi

exec "$@"