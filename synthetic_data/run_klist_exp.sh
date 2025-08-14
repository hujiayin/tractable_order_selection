#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$SCRIPT_DIR"
source "$BASE_DIR/exec_params.sh"

RELATION_TYPE="path3"
NUM_ROWS_LIST=(10 100 1000 10000 100000 1000000)
NUM_ROWS_LIST=(4 32 317 3173 31623 316227)
# NUM_ROWS_LIST=(10 100 1000 10000)
DOMAIN_TYPE="small" 
DIST_TYPE="uniform"
SEED=42
EXTRA="joine11"

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

echo "Save directory: $PATTERN_NAME"

# bash ./run_generator.sh -r "$RELATION_TYPE" -n "$(IFS=,; echo "${NUM_ROWS_LIST[*]}")" -d "$DOMAIN_TYPE" -t "$DIST_TYPE" -s "$SEED" -x "$EXTRA" -p "$PATTERN_NAME"

# bash ./run_count.sh -r "$RELATION_TYPE" -p "$PATTERN_NAME"
# bash ./run_da.sh -r "$RELATION_TYPE" -p "$PATTERN_NAME"
# bash ./run_select.sh -r "$RELATION_TYPE" -p "$PATTERN_NAME"
# bash ./run_pa.sh -r "$RELATION_TYPE" -p "$PATTERN_NAME"
bash ./run_pg.sh -r "$RELATION_TYPE" -p "$PATTERN_NAME"


