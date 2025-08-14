import argparse
import csv
import json
from pathlib import Path
from select_k.PandasAccess import PandasAccess
from exp_timer.exp_timer import CONFIG, time_block, time_all_methods, TimerConfig, set_timer_context, timer_records
from exp_utils import parse_args, append_timing_records, append_result_records

def main():
    args = parse_args()
    exp_id = getattr(args, "exp_id", "exp01")
    trial = getattr(args, "trial", 1)
    time_data_file = getattr(args, "time_data_file", "timing_log_pandas.csv")
    log_file = getattr(args, "timer_log", "exp_time_pandas.log")
    log_path = Path(log_file).resolve()

    # Initialize the timer configuration
    CONFIG.__init__(log_file=log_path, enabled=args.timer_enabled, threshold_ms=0.0)
    set_timer_context(exp_id=exp_id, trial=trial)

    # Load k_list from file if provided
    k_values = None
    k_path = Path(args.k_file).resolve()
    if k_path.exists():
        try:
            k_json = json.loads(k_path.read_text())
            if isinstance(k_json, dict) and "k_list" in k_json:
                k_values = k_json["k_list"]
            elif isinstance(k_json, list):
                k_values = k_json
        except Exception as e:
            print(f"Error when reading {k_path}: {e}")

    # if invalid k_file use args.k_list
    if not k_values:
        k_values = args.k_list

    result_list = []
    query_file = Path(args.query_file).resolve()
    data_dir = Path(args.data_dir).resolve() if hasattr(args, "data_dir") else query_file.parent

    pa = PandasAccess(query_file, data_dir=data_dir)
    pa.smart_join_and_sort()

    pa.get_result(0)  # warmup call
    
    for k in k_values:
        set_timer_context(exp_id=exp_id, trial=trial, k=k)
        result = pa.get_result(k)
        if trial == 1:
            result_list.append({
                "exp_id": exp_id,
                "k": k,
                "result": result
            })

    # append timing records to the output file
    time_data_output_path = Path(time_data_file).resolve()
    append_timing_records(timer_records, time_data_output_path)
    if trial == 1:
        records_file = getattr(args, "records_file", "records_pandas.csv")
        records_file = Path(records_file).resolve()
        append_result_records(result_list, records_file)
        


if __name__ == "__main__":
    main()
