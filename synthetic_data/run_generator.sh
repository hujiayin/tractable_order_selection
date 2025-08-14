#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$SCRIPT_DIR"
source "$BASE_DIR/exec_params.sh"

RELATION_TYPE="path3"
NUM_ROWS_LIST=(1000 1000 1000)
DOMAIN_TYPE="small"
DIST_TYPE="uniform"
SEED=42
EXTRA=""

domain_initial="${DOMAIN_TYPE:0:1}"
dist_initial="${DIST_TYPE:0:1}"
if [ -n "$DIST_PARAMS" ]; then
  dist_str="${dist_initial}${DIST_PARAMS}"
else
  dist_str="$dist_initial"
fi
if [ -n "$EXTRA" ]; then
  extra_str="_${EXTRA}"
else
  extra_str=""
fi
PATTERN_NAME="${RELATION_TYPE}_${domain_initial}_${dist_str}_${SEED}${extra_str}" 

while getopts "r:n:d:t:s:x:p:" opt; do
  case $opt in
    r) RELATION_TYPE="$OPTARG" ;;
    n) IFS=',' read -r -a NUM_ROWS_LIST <<< "$OPTARG" ;;
    d) DOMAIN_TYPE="$OPTARG" ;;
    t) DIST_TYPE="$OPTARG" ;;
    s) SEED="$OPTARG" ;;
    x) EXTRA="$OPTARG" ;;
    p) PATTERN_NAME="$OPTARG" ;;

    \?) echo "Invalid option: -$OPTARG" >&2 ;;
  esac
done

echo "Generated data_dir_name: $PATTERN_NAME"

QUERY_FILE="$BASE_DIR/$RELATION_TYPE/full_query/query.json"
GEN_DATA_FILE="$BASE_DIR/data_generator.py"
SAVE_DIR="$BASE_DIR/$RELATION_TYPE/input/$PATTERN_NAME"

echo "Save directory: $SAVE_DIR"


python $GEN_DATA_FILE \
    --query_file "$QUERY_FILE" \
    --relation_type "$RELATION_TYPE" \
    --num_rows ${NUM_ROWS_LIST[*]} \
    --domain_type "$DOMAIN_TYPE" \
    --dist_type "$DIST_TYPE" \
    --seed "$SEED" \
    --save_dir "$SAVE_DIR" \
    --extra "$EXTRA"


