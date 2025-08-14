#!/bin/bash
set -e

# -------------------------
# Parameters for the direct access experiment
# -------------------------
EXP_FILE="../../direct_access_klist.py"
QUERY_FILE="query.json"
K_FILE="k_list.json"
K_LIST=(0 1 2 3 4)
TIMER_ENABLED=true
TIMER_LOG="exp_time.log"
# -------------------------

# concatenate K_LIST into a space-separated string
K_LIST_ARGS=""
for k in "${K_LIST[@]}"; do
  K_LIST_ARGS+=" $k"
done

# run direct access experiment
python3 "$EXP_FILE" \
  --query_file "$QUERY_FILE" \
  --k_file "$K_FILE" \
  --k_list $K_LIST_ARGS \
  --timer_enabled "$TIMER_ENABLED" \
  --timer_log "$TIMER_LOG"
