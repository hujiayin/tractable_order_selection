#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$SCRIPT_DIR"

RELATION_TYPE="path3"
PATTERN_NAME="path3_n_u_123"
QUERY_NAME="full_query"
# EXP_NAME="exp1_1000"
# ITERS=1
TIMER_ENABLED=true

while getopts "r:p:" opt; do
  case $opt in
    r) RELATION_TYPE="$OPTARG" ;;
    p) PATTERN_NAME="$OPTARG" ;;

    \?) echo "Invalid option: -$OPTARG" >&2 ;;
  esac
done

QUERY_FILE="$BASE_DIR/$RELATION_TYPE/$QUERY_NAME/query.json"

INPUT_DATASETS_DIR="$BASE_DIR/$RELATION_TYPE/input/$PATTERN_NAME"

CT_EXPERIMENT_FILE="../exp_count.py"

for DATASET_DIR in "$INPUT_DATASETS_DIR"/*/; do 

  EXP_ID=$(basename "$DATASET_DIR")

  echo "[+] Running dataset: $EXP_ID"

    python3 "$CT_EXPERIMENT_FILE" \
      --query_file "$QUERY_FILE" \
      --data_dir "$DATASET_DIR" 
 
done
