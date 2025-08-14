#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$SCRIPT_DIR"
source "$BASE_DIR/exec_params.sh"

RELATION_TYPE="path3"
PATTERN_NAME="path3_n_u_123"
QUERY_NAME="full_query"
# EXP_NAME="exp1_1000"
# ITERS=1
TIMER_ENABLED=true

while getopts "r:p:q:e:i:" opt; do
  case $opt in
    r) RELATION_TYPE="$OPTARG" ;;
    p) PATTERN_NAME="$OPTARG" ;;
    q) QUERY_NAME="$OPTARG" ;;
    e) EXP_NAME="$OPTARG" ;;
    i) ITERS="$OPTARG" ;;

    \?) echo "Invalid option: -$OPTARG" >&2 ;;
  esac
done

QUERY_FILE="$BASE_DIR/$RELATION_TYPE/$QUERY_NAME/query.json"

INPUT_DATASETS_DIR="$BASE_DIR/$RELATION_TYPE/input/$PATTERN_NAME"
OUTPUT_DIR="$BASE_DIR/$RELATION_TYPE/$QUERY_NAME/$PATTERN_NAME"
echo "Output directory: $OUTPUT_DIR"

# output directory
mkdir -p "$OUTPUT_DIR"

SLT_EXPERIMENT_FILE="../select_klist.py"

TIMER_LOG="$OUTPUT_DIR/timing_log_select.log"
TIME_DATA_FILE="$OUTPUT_DIR/timing_log_select.csv"
RECORDS_FILE="$OUTPUT_DIR/records_select.csv"

for f in "$TIMER_LOG" "$TIME_DATA_FILE" "$RECORDS_FILE"; do
  if [ -f "$f" ]; then
    echo "Deleting existing file: $f"
    rm -f "$f"
  fi
done


echo "Starting batch experiments..." > "$TIMER_LOG"

for DATASET_DIR in "$INPUT_DATASETS_DIR"/*/; do 

 
  EXP_ID=$(basename "$DATASET_DIR")
  K_FILE="$DATASET_DIR/k_list.json"

  echo "[+] Running dataset: $EXP_ID"

  for i in $(seq 1 $ITERS); do
    echo " -------- Trial $i"

    python3 "$SLT_EXPERIMENT_FILE" \
      --query_file "$QUERY_FILE" \
      --data_dir "$DATASET_DIR" \
      --k_file "$K_FILE" \
      --timer_enabled True \
      --timer_log "$TIMER_LOG" \
      --time_data_file "$TIME_DATA_FILE" \
      --trial "$i" \
      --exp_id "$EXP_ID" \
      --records_file "$RECORDS_FILE"
  done
done