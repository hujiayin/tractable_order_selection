# %%
from select_k.Query import ConjunctiveQuery
from select_k.Selection_Sum import Selection_Sum
import pandas as pd
# %%
def selection_by_sum(atoms, free_vars, sum_order, data, k): 
    cq = ConjunctiveQuery(atoms, free_vars, data = data, sum_order = sum_order)
    select_cq = Selection_Sum(cq)
    result = select_cq.select_k(k)
    print(f'[{k}]: {result}')
    return result

# pandas dataframe result: for reference
def smart_join_and_sort(atoms, data, sum_order, free_variables):
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

    if isinstance(sum_order, list):
        sort_cols = [v for v in sum_order if v in df_result.columns]
        df_result['sum'] = df_result[sort_cols].sum(axis=1)
        df_result = df_result.sort_values(by='sum').drop(columns='sum').reset_index(drop=True)
    elif isinstance(sum_order, dict):
        sort_cols = [v for v in sum_order if v in df_result.columns]
        df_result['weighted_sum'] = df_result[list(sum_order.keys())].mul(list(sum_order.values()), axis=1).sum(axis=1)
        df_result = df_result.sort_values(by='weighted_sum').drop(columns='weighted_sum').reset_index(drop=True)
    else:
        # If sum_order is not a list or dict, throw error
        raise ValueError("Invalid sum_order format")

    leading_cols = [col for col in free_variables if col in df_result.columns]
    # remaining_cols = [col for col in df_result.columns if col not in leading_cols]
    return df_result[leading_cols].drop_duplicates().reset_index(drop=True)

def test_s(): 
    df_result = smart_join_and_sort(atoms, data, sum_order, free_vars)
    print(df_result)

    df_list = df_result.to_dict(orient='records') 
    df_list = [{k: v for k, v in d.items() if k != 'index'} for d in df_list]
    flag = 1
    print_res = ''
    for i, row_dict in enumerate(df_list): 
        print('K index: ', i, " : ", row_dict)
        selection_res = selection_by_sum(atoms, free_vars, sum_order, data, i) 
        if row_dict != selection_res:
            # Check if the sum is the same (rearrangement because of ties)
            if not (row_dict['weighted_sum'] == selection_res['weighted_sum']):
                flag = 0 
                print_res += f"Mismatch at row {i}:\n  correct:  {row_dict}\n  selection: {selection_res}\n"
    print(print_res)
    if flag: 
        print("PASS ALL")
        
# %% 
"""
sum_order format
1. List: sort by the sum of variables in the list in ascending order
2. Dict: sort by the sum of variables in the list in ascending order, each one multiplied with the value in the dict
"""

# %%
# 2 relations. Full acyclic CQ
free_vars = ['a', 'b', 'c']
atoms = [('P', ('a', 'b')), ('Q', ('b', 'c'))]

data = {
    'P': {
        'a': [1, 5, 2],
        'b': [1, 2, 1],
    },
    'Q': {
        'b': [2, 1, 2],
        'c': [1, 3, 2],
    },
}
sum_order = ['a', 'b', 'c']
# sum_order = {'c':-1, 'a':1, }
test_s()

# %%
# Non-full CQ
# atoms = [
#     ('A', ('u', 'v', 'x')),       # A(u, v, x)
#     ('B', ('v', 'w', 'y')),       # B(v, w, y) -- join on v
#     ('C', ('x', 'z')),            # C(x, z) -- join on x
#     ('D', ('p', 'q')),            # D(p, q) -- Cartesian join (no shared vars)
# ]

# data = {
#     'A': {
#         'u': ['u1', 'u2', 'u3', 'u1', 'u2'],
#         'v': ['v1', 'v2', 'v1', 'v2', 'v3'],
#         'x': ['x1', 'x2', 'x3', 'x1', 'x4'],
#     },
#     'B': {
#         'v': ['v1', 'v1', 'v2', 'v3', 'v2'],
#         'w': ['w1', 'w2', 'w1', 'w3', 'w2'],
#         'y': ['y1', 'y2', 'y3', 'y4', 'y5'],
#     },
#     'C': {
#         'x': ['x1', 'x2', 'x3', 'x2', 'x1'],
#         'z': ['z1', 'z2', 'z3', 'z4', 'z5'],
#     },
#     'D': {
#         'p': [1, 2, 3],
#         'q': ['q2', 'q1', 'q3'],
#     }
# }
# lex_order = ['u', 'v', 'x', 'y', 'z']
# free_vars = ['u', 'v', 'x', 'y', 'z']
# test_s()

# %%
"""
Test for single k
"""
# k = 12
# result = selection(atoms, free_vars, lex_order, data, k)
# %%
