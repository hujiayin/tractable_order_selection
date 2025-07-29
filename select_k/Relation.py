# import numpy as np
from typing import Dict, Tuple, Optional, List

class Relation:
    def __init__(self, name: str, 
                 variables: List[str]|Tuple[str], 
                 instance: Optional[List[Tuple]|Dict[str, List]] = None,
                 lex_order: Optional[List|Dict[str, int]]=None, 
                 connection: Optional[List] = None, 
                 need_check: bool=True, ):
        self.name = name
        self.variables = tuple(variables)
        self.width = len(self.variables)
        self.lex_vars = None # list
        self.lex_dict = None # dict

        if lex_order: 
            self.add_lexorder(lex_order)
            
        # self.instance_col = None
        self.instance_row = None 
        self.rowid = None
        
        if instance: 
            self.add_instance(instance, need_check)

        self.connection = connection if connection else [] # connection to the original non-projected relation 
    
    def __repr__(self):
        return f"------Relation {self.name} {self.variables}\n{self.lex_vars} {self.lex_dict}\
                \n{self.connection}\n{self.rowid}\n{self.instance_row}\n---------"
    

    # Add lex_order
    def add_lexorder(self, lex_order: List|Dict[str, int]): 
        self.lex_vars = {v: self.variables.index(v) for v in lex_order if v in self.variables}
        if isinstance(lex_order, List):  # ascending by default
            self.lex_dict = {v: 1 for v in lex_order if v in self.variables} 
        elif isinstance(lex_order, Dict): # by the given ranking type
            self.lex_dict = {v: lex_order[v] for v in lex_order if v in self.variables} 

    # def update_lexorder(self, query_lex_dict, conn_vars): 
    #     self.lex_vars = {v: self.variables.index(v) for v in query_lex_dict if v in self.variables} 

    #     self.lex_dict = {v: (1 if (query_lex_dict[v] == 0 and v in conn_vars) else query_lex_dict[v]) \
    #                       for v in query_lex_dict if v in self.variables} 


    # Add instance
    def add_instance(self, data: List[Tuple]|Dict[str, List], need_check: bool=True): 

        # validate variables and same count of rows for every variable
        if isinstance(data, Dict): 
            len_list = [len(v) for v in data.values()] 
            data_len = len_list[0]
            if need_check:
                if set(data.keys()) != set(self.variables): 
                    raise Exception(f'Relation {self.name}: Invalid Instance. Variables do not match!')
                
                flag = len(len_list) > 0 and all(x == len_list[0] for x in len_list) 
                if not flag: 
                    raise Exception(f'Relation {self.name}: Invalid Instance. Rows do not match!')
                # self.instance_col = {col: data[col] for col in self.variables}
            self.instance_row = list(zip(*{col: data[col] for col in self.variables}.values()))
        elif isinstance(data, List): 
            data_len = len(data) # number of rows
            if need_check:
                var_num = len(self.variables) # number of columns
                flag = all(len(x) == var_num for x in data) 
                if not flag: 
                    raise Exception(f'Relation {self.name}: Invalid Instance. Data does not match!') 
            # data_t = list(zip(*data))
            # self.instance_col = {col: list(vals) for col, vals in zip(self.variables, data_t)}
            self.instance_row = data
            
        self.rowid = list(range(data_len)) # initial rowid
         


    def semi_join(self, child_rel: "Relation", join_attrs: List[str]): 
        # column data to tuple
        child_attr_map = [child_rel.variables.index(attr) for attr in join_attrs]
        keyset = set(tuple(record[attr] for attr in child_attr_map) for record in child_rel.instance_row)
        current_attr_map = [self.variables.index(attr) for attr in join_attrs]
        target_keys = [tuple(record[attr] for attr in current_attr_map) for record in self.instance_row]
        mask = [k in keyset for k in target_keys] 
        self.instance_row = [record for record, keep in zip(self.instance_row, mask) if keep]
        
        self.rowid = list(range(len(self.instance_row)))

    @staticmethod
    def project_remove_duplicates(data: List[Tuple], orig_vars: List[str], new_vars: List[str]):
        """
        project and remove duplicates
        """
        new_data = []

        if new_vars: 
            # project to corresponding columns
            indices = [orig_vars.index(c) for c in new_vars] 
            new_data = list(dict.fromkeys(tuple(row[i] for i in indices) for row in data)) # keep the original order (we CARE!!! the order here)


        return new_data
    
    def lex_sort(self, query_lex_dict): 
        lex_dict = {v: query_lex_dict[v] for v in query_lex_dict if v in self.variables} 
        self.lex_vars = {v: self.variables.index(v) for v in query_lex_dict if v in self.variables}
        # lex_vars = self.lex_vars 
        # lex_dict = self.lex_dict
        for var, col_idx in reversed(self.lex_vars.items()):
            if lex_dict[var] != 0:
                self.rowid.sort(key=lambda i: self.instance_row[i][col_idx], reverse=(lex_dict[var] == -1)) 

