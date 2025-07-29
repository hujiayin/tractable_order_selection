# %%
from select_k.Query import ConjunctiveQuery
from select_k.Selection import Selection
import pandas as pd
# %%
def selection(atoms, free_vars, lex_order, data, k): 
    cq = ConjunctiveQuery(atoms, free_vars, lex_order, data=data)
    select_cq = Selection(cq)
    result = select_cq.select_k(k)
    print(f'[{k}]: {result}')
    return result

# pandas dataframe result: for reference
def smart_join_and_sort(atoms, data, lex_order, free_varaibles):
    df_result = None
    for rel_name, rel_vars in atoms:

        df_current = pd.DataFrame({var: data[rel_name][var] for var in rel_vars})

        if df_result is None:
            df_result = df_current
        else:
            common_vars = list(set(df_result.columns).intersection(set(df_current.columns)))
            if common_vars:
                df_result = df_result.merge(df_current, on=common_vars, how='inner')
            else:
                df_result['__tmp_key'] = 1
                df_current['__tmp_key'] = 1
                df_result = df_result.merge(df_current, on='__tmp_key').drop(columns='__tmp_key')

    if isinstance(lex_order, list):
        sort_cols = [v for v in lex_order if v in df_result.columns]
        df_result = df_result.sort_values(by=sort_cols).reset_index(drop=True)
    else: 
        sort_cols = [v for v in list(lex_order.keys()) if v in df_result.columns] 
        sort_type = [(lex_order[v]==1) for v in list(lex_order.keys()) if v in df_result.columns] 
        df_result = df_result.sort_values(by=sort_cols, ascending=sort_type).reset_index(drop=True)
    leading_cols = [col for col in free_varaibles if col in df_result.columns]
    remaining_cols = [col for col in df_result.columns if col not in leading_cols]
    return df_result[leading_cols].drop_duplicates().reset_index(drop=True)

def test_s(): 
    df_result = smart_join_and_sort(atoms, data, lex_order, free_vars)
    print(df_result)
    # selection_res = selection(atoms, free_vars, lex_order, data, k) 

    df_list = df_result.to_dict(orient='records') 
    df_list = [{k: v for k, v in d.items() if k != 'index'} for d in df_list]
    flag = 1
    print_res = ''
    for i, row_dict in enumerate(df_list): 
        print('K index: ', i)
        selection_res = selection(atoms, free_vars, lex_order, data, i) 
        if row_dict != selection_res:
            flag = 0 
            print_res += f"Mismatch at row {i}:\n  correct:  {row_dict}\n  selection: {selection_res}\n"
    print(print_res)
    if flag: 
        print("PASS ALL")
        
# %% 
"""
lex_order format
1. List: all the variables in list will be sorted in ascending order
2. Dict: all the variables in list will be sorted according to the corresponding value
        value 
        [   1 - ascending; 
            -1 - decending; 
        ]
"""

# %%
# 2 relations. Full acyclic CQ, complete/partial lex_order
free_vars = ['a', 'b', 'c']
atoms = [('P', ('b', 'a')), ('Q', ('c', 'b'))]

data = {
    'P': {
        'a': ['x1', 'x3', 'x2'],
        'b': ['y1', 'y2', 'y1'],
    },
    'Q': {
        'b': ['y2', 'y1', 'y2'],
        'c': ['z1', 'z3', 'z2'],
    },
}
# lex_order = ['c', 'a', 'b']
lex_order = {'c':-1, 'a':1, }
test_s()

# %%
# Non-full CQ
atoms = [
    ('A', ('u', 'v', 'x')),       # A(u, v, x)
    ('B', ('v', 'w', 'y')),       # B(v, w, y) -- join on v
    ('C', ('x', 'z')),            # C(x, z) -- join on x
    ('D', ('p', 'q')),            # D(p, q) -- Cartesian join (no shared vars)
]

data = {
    'A': {
        'u': ['u1', 'u2', 'u3', 'u1', 'u2'],
        'v': ['v1', 'v2', 'v1', 'v2', 'v3'],
        'x': ['x1', 'x2', 'x3', 'x1', 'x4'],
    },
    'B': {
        'v': ['v1', 'v1', 'v2', 'v3', 'v2'],
        'w': ['w1', 'w2', 'w1', 'w3', 'w2'],
        'y': ['y1', 'y2', 'y3', 'y4', 'y5'],
    },
    'C': {
        'x': ['x1', 'x2', 'x3', 'x2', 'x1'],
        'z': ['z1', 'z2', 'z3', 'z4', 'z5'],
    },
    'D': {
        'p': [1, 2, 3],
        'q': ['q2', 'q1', 'q3'],
    }
}
lex_order = ['u', 'v', 'x', 'y', 'z']
free_vars = ['u', 'v', 'x', 'y', 'z']
test_s()

# %%
"""
Test for single k
"""
# k = 12
# result = selection(atoms, free_vars, lex_order, data, k)
# %%
