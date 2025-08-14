import argparse
from pathlib import Path
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="Run query experiment.")
    parser.add_argument(
        "--query_file",
        type=str,
        # default="query.json",
        help="Path to the query JSON file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        # default="query.json",
        help="Path to the data directory"
    )
    parser.add_argument(
        "--k_file",
        type=str,
        # default="k_list.json",
        help="Path to the k JSON file"
    )
    parser.add_argument(
        "--k_list",
        type=int,
        nargs="*",
        default=[0],
        help="List of k values to use (used if k_file is missing or invalid)"
    )
    parser.add_argument(
        "--timer_enabled",
        type=bool,
        default=False,
        help="Enable or disable the timer"
    )
    parser.add_argument(
        "--timer_log",
        type=str,
        default="exp_time.log",
        help="Path to the timer log file"
    )
    parser.add_argument(
        "--exp_id",
        type=str,
        default="exp01",
        help="Experiment ID"
    )
    parser.add_argument(
        "--trial",
        type=int,
        # default=1,
        help="Trial number for the experiment"
    )
    parser.add_argument(
        "--time_data_file",
        type=str,
        # default="timing_log.csv",
        help="Path to the timing data CSV file"
    )
    parser.add_argument(
        "--records_file",
        type=str,
        # default="records.csv",
        help="Path to the records CSV file"
    )
    return parser.parse_args()

def append_timing_records(timer_records: list[dict], file_path: Path):
    """Append timing records to a CSV file.
    If the file does not exist, it will be created with a header.
    If the file exists but is empty, it will also write a header.
    """
    file_path = Path(file_path).resolve()
    write_header = not file_path.exists() or file_path.stat().st_size == 0

    fieldnames = ["timestamp", "exp_id", "trial", "process", "k", "duration_ms"]

    with file_path.open("a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(timer_records)

def append_result_records(result_list: list[dict], file_path: Path):
    """Append result records to a CSV file.
    """
    write_header = not file_path.exists() or file_path.stat().st_size == 0

    fieldnames = ["exp_id", "k", "result"]

    with file_path.open("a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(result_list)

