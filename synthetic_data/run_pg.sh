#!/bin/bash
set -e
source pg_config.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$SCRIPT_DIR"
source "$BASE_DIR/exec_params.sh"

RELATION_TYPE="path3"
PATTERN_NAME="path3_n_u_123"
QUERY_NAME="full_query"
# EXP_NAME="exp1_1000"
ITERS=3
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
QUERY_DIR="$BASE_DIR/$RELATION_TYPE/$QUERY_NAME"

INPUT_DATASETS_DIR="$BASE_DIR/$RELATION_TYPE/input/$PATTERN_NAME"
OUTPUT_DIR="$BASE_DIR/$RELATION_TYPE/$QUERY_NAME/$PATTERN_NAME"
echo "Output directory: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

echo "Generate SQL files..."
GEN_SQL_FILE="../query_to_sql.py"
python $GEN_SQL_FILE --query_file "$QUERY_FILE" 

PROCESS_NAME="pg_query"

echo "exp_id,k,result" > "$OUTPUT_DIR/records_pg.csv"
echo "timestamp,exp_id,trial,process,k,duration_ms" > "$OUTPUT_DIR/timing_log_pg.csv"

for DATASET_DIR in "$INPUT_DATASETS_DIR"/*/; do 
  
  EXP_ID=$(basename "$DATASET_DIR")
  K_FILE="$DATASET_DIR/k_list.json"

  DB_NAME="${PATTERN_NAME}_${EXP_ID}"

  echo "[Step 2] Drop and recreate database: $DB_NAME..."
  dropdb --if-exists "$DB_NAME" -U "$PG_DB_USER"
  createdb "$DB_NAME" -U "$PG_DB_USER"

  cd "$DATASET_DIR"

  echo "[Step 3] Create tables and import data..."
  psql -U "$PG_DB_USER" -d "$DB_NAME" -f "$QUERY_DIR/create_tables.sql" 

  cd "$BASE_DIR"

  echo "[Step 4] Load k list..."
  K_LIST=$(jq -r '.[]' "$K_FILE")

  echo "[Step 5] Run queries and explanations..."

  for k in $K_LIST; do
    echo "Running for k=$k"
    
    # Run query
    SQL=$(sed "s/{k}/$k/" "$QUERY_DIR/query_template.sql")
    echo "Executing SQL: $SQL"
    VALUE=$(psql -U "$PG_DB_USER" -d "$DB_NAME" -t -c "$SQL" | tr -d '[:space:]')
    echo "$EXP_ID,$k,$VALUE" >> "$OUTPUT_DIR/records_pg.csv"

    # Run explain multiple times
    for ((i=1; i<=$ITERS; i++)); do
      SQL_EXPLAIN=$(sed "s/{k}/$k/" "$QUERY_DIR/explain_template.sql")

      if [ "$i" -eq 1 ]; then
        EXPLAIN_OUTPUT_DIR="$OUTPUT_DIR/$EXP_ID"
        mkdir -p "$EXPLAIN_OUTPUT_DIR"
        EXPLAIN_OUTPUT_FILE="$EXPLAIN_OUTPUT_DIR/explain_offset${k}.txt"
        psql -U "$PG_DB_USER" -d "$DB_NAME" -c "$SQL_EXPLAIN" > "$EXPLAIN_OUTPUT_FILE"
        TIME_MS=$(grep "Execution Time" "$EXPLAIN_OUTPUT_FILE" | awk '{print $3}')
      else
        TIME_MS=$(psql -U "$PG_DB_USER" -d "$DB_NAME" -t -c "$SQL_EXPLAIN" \
          | grep "Execution Time" | awk '{print $3}')
      fi  
      echo "$(date +%s%3N),$EXP_ID,$i,$PROCESS_NAME,$k,$TIME_MS" >> "$OUTPUT_DIR/timing_log_pg.csv"
    done
  done

  dropdb --if-exists "$DB_NAME" -U "$PG_DB_USER"

done

echo "All done. Results saved to $OUTPUT_DIR"
