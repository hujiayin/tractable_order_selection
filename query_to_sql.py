# query_to_sql.py
import json
import sys
from pathlib import Path
import argparse

def main():

    parser = argparse.ArgumentParser(description="Convert query JSON to SQL")
    parser.add_argument("--query_file", type=str, help="Path to the query JSON file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for SQL files")
    args = parser.parse_args()

    query_file = Path(args.query_file).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(query_file).parent

    with open(query_file) as f:
        config = json.load(f)

    query = config["query"]
    free_vars = config["free_variables"]
    lex_order = config["lex_order"]

    create_stmts = []
    copy_stmts = []
    join_stmts = []
    tables = []

    for rel in query:
        name = rel["relation_name"]
        schema = rel["relation_schema"]
        short_cols = [f"{col}" for col in schema]
        full_cols = [f"{name}.{col}" for col in schema]
        joined_schema = ", ".join(f"{col} INT" for col in short_cols)
        create_stmts.append(f'CREATE TABLE {name} ({joined_schema});')

        copy_stmts.append(
            f"\\copy {name} FROM '{rel['file_name']}' DELIMITER ',' CSV;"
        )
        tables.append(name)

    # Join conditions
    join_query = f"FROM {tables[0]}\n"
    for rel in query[1:]:
        name = rel["relation_name"]
        conds = rel.get("join_condition", [])
        if conds:
            on_expr = " AND ".join(conds)
            join_query += f"JOIN {name} ON {on_expr}\n"
        else:
            join_query += f"CROSS JOIN {name}\n"

    select_clause = ", ".join(f'{var} AS "{var}"' for var in free_vars)

    # Build ORDER BY
    order_clause = ", ".join(f"{k} {'ASC' if v == 1 else 'DESC'}" for k, v in lex_order.items())

    with open(f"{output_dir}/create_tables.sql", "w") as f:
        
        for stmt in create_stmts:
            f.write(stmt + "\n")
        for stmt in copy_stmts:
            f.write(stmt + "\n")
        f.write("VACUUM ANALYZE;\n")

    with open(f"{output_dir}/query_template.sql", "w") as f:
        f.write(f"SELECT {select_clause}\n{join_query}ORDER BY {order_clause} OFFSET {{k}} LIMIT 1;\n")

    with open(f"{output_dir}/explain_template.sql", "w") as f:
        f.write(f"EXPLAIN (ANALYZE, BUFFERS)\nSELECT {select_clause}\n{join_query}ORDER BY {order_clause} OFFSET {{k}} LIMIT 1;\n")

if __name__ == "__main__":
    main()