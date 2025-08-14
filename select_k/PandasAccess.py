import json
import pandas as pd
from typing import Union
from exp_timer.exp_timer import timer
from pathlib import Path

class PandasAccess:
    @timer(name="LoadQuery", extra=lambda ctx: f"exp={ctx.exp_id}_trial={ctx.trial}_k={ctx.k}" if hasattr(ctx, 'exp_id') and hasattr(ctx, 'trial') and hasattr(ctx, 'k') else None)
    def __init__(self, query_file: Path, data_dir:Path=None):
        with query_file.open('r') as f:
            query_data = json.load(f)

        self.query = query_data["query"]
        self.free_variables = query_data["free_variables"]
        self.lex_order = query_data["lex_order"]
        data_dir = data_dir if data_dir else query_file.parent
        self.data_dfs = self._load_dataframes(data_dir=data_dir)
        self.sorted_df = None

    def _load_dataframes(self, data_dir:Path=None):
        data_dfs = {}
        for rel in self.query:
            rel_name = rel["relation_name"]
            rel_schema = rel["relation_schema"]
            file_path = data_dir / rel["file_name"]
            has_header = rel.get("has_header", False)
            header = 0 if has_header else None
            df = pd.read_csv(file_path, header=header)
            if not has_header:
                df.columns = [f"{rel_name}.{attr}" for attr in rel_schema]
            data_dfs[rel_name] = df
        return data_dfs

    def smart_join_and_sort(self):
        df_result = self.smart_join()
        self.sorted_df = self.sort_joined(df_result)

    @timer(name="SmartJoin", extra=lambda ctx: f"exp={ctx.exp_id}_trial={ctx.trial}_k={ctx.k}" if hasattr(ctx, 'exp_id') and hasattr(ctx, 'trial') and hasattr(ctx, 'k') else None)
    def smart_join(self): 
        df_result = None
        for rel in self.query:
            rel_name = rel["relation_name"]
            rel_df = self.data_dfs[rel_name]
            join_conds = rel.get("join_condition", [])

            if df_result is None:
                df_result = rel_df
                continue

            joined = False
            for cond in join_conds:
                left, right = [x.strip() for x in cond.split("=")]
                if left in df_result.columns and right in rel_df.columns:
                    df_result = df_result.merge(rel_df, left_on=left, right_on=right, how="inner")
                    joined = True
                    break
                elif right in df_result.columns and left in rel_df.columns:
                    df_result = df_result.merge(rel_df, left_on=right, right_on=left, how="inner")
                    joined = True
                    break

            if not joined:
                # Cartesian product
                df_result['_tmp_key'] = 1
                rel_df['_tmp_key'] = 1
                df_result = df_result.merge(rel_df, on='_tmp_key').drop(columns='_tmp_key')
        return df_result
    
    @timer(name="Sort", extra=lambda ctx: f"exp={ctx.exp_id}_trial={ctx.trial}_k={ctx.k}" if hasattr(ctx, 'exp_id') and hasattr(ctx, 'trial') and hasattr(ctx, 'k') else None)
    def sort_joined(self, df_result: pd.DataFrame):
        if isinstance(self.lex_order, list):
            sort_cols = [v for v in self.lex_order if v in df_result.columns]
            df_result = df_result.sort_values(by=sort_cols).reset_index(drop=True)
        else:
            sort_cols = [v for v in list(self.lex_order.keys()) if v in df_result.columns]
            sort_type = [(self.lex_order[v] == 1) for v in list(self.lex_order.keys()) if v in df_result.columns]
            df_result = df_result.sort_values(by=sort_cols, ascending=sort_type).reset_index(drop=True) 
        leading_cols = [col for col in self.free_variables if col in df_result.columns]
        return df_result[leading_cols].drop_duplicates().reset_index(drop=True)

    
    @timer(name="GetResult", extra=lambda ctx: f"exp={ctx.exp_id}_trial={ctx.trial}_k={ctx.k}" if hasattr(ctx, 'exp_id') and hasattr(ctx, 'trial') and hasattr(ctx, 'k') else None)
    def get_result(self, k: int = 0):
        return self.sorted_df.iloc[k].to_dict()

