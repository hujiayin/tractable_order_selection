import argparse
import os
from pathlib import Path
import random
import numpy as np
from math import sqrt
import pandas as pd
from datetime import datetime
import json
import csv

def sample_value(domain_values, dist_type, dist_params):
    if dist_type == 'uniform':
        return random.choice(domain_values)
    
    elif dist_type == 'zipf':
        while True:
            zipf_param = dist_params[0]
            i = np.random.zipf(zipf_param) - 1
            if i < len(domain_values):
                return domain_values[i]
            
    elif dist_type == 'normal':
        sigma_factor = dist_params[0] if dist_params else 6
        mu = len(domain_values) / 2
        sigma = len(domain_values) / sigma_factor
        for _ in range(10):
            i = int(np.random.normal(mu, sigma))
            if 0 <= i < len(domain_values):
                return domain_values[i]
        return random.choice(domain_values)
    
    else:
        raise ValueError(f"Unsupported dist_type: {dist_type}")

def generate_relation_data(variables, domain_values, num_rows, dist_type, dist_params):
    seen = set()
    rel_data = []
    attempts = 0
    max_attempts = 10 * num_rows

    while len(rel_data) < num_rows and attempts < max_attempts:
        
        tup = tuple(
            sample_value(domain_values, dist_type, dist_params)
            for var in variables
        )
        if tup not in seen:
            seen.add(tup)
            rel_data.append(tup)
        attempts += 1

    return rel_data

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data based on given parameters.")
    parser.add_argument("--query_file", type=str, default="query.json", help="Path to the query JSON file")
    parser.add_argument("--relation_type", type=str, help="Type of the relation to generate")
    parser.add_argument("--num_rows", type=int, nargs="*", required=True, help="Number of rows per relation.")
    parser.add_argument("--domain_type", type=str, required=True, help="Dictionary of domain sizes per variable in JSON format.")
    parser.add_argument("--dist_type", type=str, required=True, help="Dictionary of distribution types and parameters in JSON format.") 
    parser.add_argument("--dist_params", type=int, nargs="*", required=False, help="Parameters for the distribution types, if applicable.")
    parser.add_argument("--save_dir", type=str, required=False, help="Directory to save the generated data).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--extra", type=str, default='', help="Extra identifier for data generation.")
    args = parser.parse_args()

    # Load query configuration
    json_path = Path(args.query_file).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file {json_path} does not exist.")


    json_obj = json.loads(json_path.read_text())
    relations = json_obj["query"]
    relation_type = args.relation_type if args.relation_type else json_path.parent.parent.name
    num_rows_list = args.num_rows 
    domain_type = args.domain_type
    dist_type = args.dist_type
    dist_params = args.dist_params if args.dist_params else []
    seed = args.seed 
    dist_str = f'{dist_type[0]}{dist_params}' if dist_params else dist_type[0]
    extra_str = f'_{args.extra}' if args.extra else ''
    data_dir_name = f'{relation_type}_{domain_type[0]}_{dist_str}_{seed}{extra_str}'
    data_dir = Path(args.save_dir).resolve() if args.save_dir else json_path.parent / data_dir_name
    if not data_dir.exists():
        os.makedirs(data_dir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    for i, num_row in enumerate(num_rows_list): 
        
        data_dir_i = data_dir / f"exp{i+1}_{num_row}"
        if not data_dir_i.exists():
            os.makedirs(data_dir_i, exist_ok=True)
        generate_data(
                                        relations=relations,
                                        num_row=num_row,
                                        domain_type=domain_type,
                                        dist_type=dist_type,
                                        dist_params=dist_params,
                                        data_dir=data_dir_i
                                    )
        

def generate_data(relations, num_row, domain_type, dist_type, dist_params, data_dir):
    if domain_type == "large": 
        domain_size = num_row
    elif domain_type == "small": 
        domain_size = sqrt(num_row) + 1 
    else: 
        raise ValueError("Not supported domain type.")

    domain_values = [i for i in range(int(domain_size))]

    for rel in relations:
        rel_name = rel["relation_name"]
        rel_schema = rel["relation_schema"]
        has_header = rel.get("has_header", False)
        file_path = data_dir / rel["file_name"]

        rel_data = generate_relation_data(rel_schema, domain_values, num_row, dist_type, dist_params)

        # Save the relation data to a CSV file

        with file_path.open("w", newline='') as f:
            writer = csv.writer(f)
            if has_header:
                writer.writerow(rel_schema)
            writer.writerows(rel_data)

if __name__ == "__main__":
    main()

# Call main function
# python synthetic_data/data_generator.py --query_file synthetic_data/input/path3_full_smallds/full_query.json --num_rows 1000 1000 1000 --domain_type 'n_row' --dist_type 'uniform' --seed 123