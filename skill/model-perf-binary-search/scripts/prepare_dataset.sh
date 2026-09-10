#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: prepare_dataset.sh SHARED_DATA_DIR DATASET_SRC" >&2
  exit 2
fi

SHARED_DATA_DIR="$1"
SELECTED_DATASET_SRC="$2"

[ -d "$SHARED_DATA_DIR" ] || {
  echo "shared dataset directory is missing: $SHARED_DATA_DIR" >&2
  exit 1
}

case "$SELECTED_DATASET_SRC" in
  /*) ;;
  *)
    echo "dataset source must be an absolute path: $SELECTED_DATASET_SRC" >&2
    exit 1
    ;;
esac

if [ ! -f "$SELECTED_DATASET_SRC" ] || [ ! -s "$SELECTED_DATASET_SRC" ]; then
  echo "dataset is missing or empty: $SELECTED_DATASET_SRC" >&2
  exit 1
fi

SHARED_DATA_DIR="$(realpath "$SHARED_DATA_DIR")"
DATASET="$(realpath "$SELECTED_DATASET_SRC")"
case "$DATASET" in
  "$SHARED_DATA_DIR"/*) ;;
  *)
    echo "dataset must be under $SHARED_DATA_DIR: $DATASET" >&2
    exit 1
    ;;
esac

printf 'DATASET=%q\n' "$DATASET"
