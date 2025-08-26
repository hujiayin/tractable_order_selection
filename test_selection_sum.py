# %%
from select_k.Query import ConjunctiveQuery
from select_k.Selection_Sum import Selection_Sum
import pandas as pd
import itertools
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
        df_result['sum'] = df_result[list(sum_order.keys())].mul(list(sum_order.values()), axis=1).sum(axis=1)
        df_result = df_result.sort_values(by='sum')
    else:
        # If sum_order is not a list or dict, throw error
        raise ValueError("Invalid sum_order format")

    leading_cols = [col for col in free_variables if col in df_result.columns]
    # remaining_cols = [col for col in df_result.columns if col not in leading_cols]
    return df_result[leading_cols].drop_duplicates().reset_index(drop=True)

def test_s(): 
    df_result = smart_join_and_sort(atoms, data, sum_order, free_vars)

    if isinstance(sum_order, list):
        sum_cols = [v for v in sum_order if v in df_result.columns]
        df_result['sum'] = df_result[sum_cols].sum(axis=1)
    elif isinstance(sum_order, dict):
        df_result['sum'] = df_result[list(sum_order.keys())].mul(list(sum_order.values()), axis=1).sum(axis=1)
    else:
        # If sum_order is not a list or dict, throw error
        raise ValueError("Invalid sum_order format")

    print(df_result)

    df_list = df_result.to_dict(orient='records') 
    df_list = [{k: v for k, v in d.items() if k != 'index'} for d in df_list]
    flag = 1
    for i, row_dict in enumerate(df_list): 
        print('K index: ', i, " : ", row_dict)
        selection_res = selection_by_sum(atoms, free_vars, sum_order, data, i) 
        if row_dict != selection_res:
            # Check if the sum is the same (rearrangement because of ties)
            if not (row_dict['sum'] == selection_res['sum']):
                flag = 0 
                print(f"Mismatch at row {i}:\n  correct:  {row_dict}\n  selection: {selection_res}\n")
                return
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
# sum_order = ['a', 'b', 'c']
sum_order = {'c':-1, 'a':1, }
k = 3 # Expected answer: {'a': 5, 'b': 2, 'c': 3}
print("Answer returned for k= ", k, ": ", selection_by_sum(atoms, free_vars, sum_order, data, k))


# %%
# 2 relations with more variables. Full acyclic CQ
free_vars = ['a', 'b', 'c', 'd', 'e']
atoms = [('P', ('a', 'b', 'c')), ('Q', ('c', 'b', 'd', 'e'))]

a_list = range(1, 10)
b_list_1 = range(1, 10)
c_list_1 = range(1, 10)

c_list_2 = range(3, 13)
b_list_2 = range(3, 13)
d_list = range(3, 13)
e_list = range(3, 13)

data = {
    'P': {
        'a': a_list,
        'b': b_list_1,
        'c': c_list_1,
    },
    'Q': {
        'b': b_list_2,
        'c': c_list_2,
        'd': d_list,
        'e': e_list,
    },
}
sum_order = ['a', 'b', 'c', 'd', 'e']
# sum_order = {'c':-1, 'a':1, }
test_s()


# %%
# 2 relations with more complex data. Full acyclic CQ
free_vars = ['a', 'b', 'c']
atoms = [('P', ('a', 'b')), ('Q', ('b', 'c'))]

# [(2, 3), (3, 2), (3, 3), (3, 4), (4, 3)]
# [(3, 3), (3, 4), (3, 5), (4, 5), (5, 5)]
a_list =   [2, 3, 3, 3, 4]
b_list_1 = [3, 2, 3, 4, 3]
b_list_2 = [3, 3, 3, 4, 5]
c_list =   [3, 4, 5, 5, 5]

data = {
    'P': {
        'a': a_list,
        'b': b_list_1,
    },
    'Q': {
        'b': b_list_2,
        'c': c_list,
    },
}
sum_order = ['a', 'b', 'c']
test_s()


# %%
# 2 relations with more complex data. Full acyclic CQ
free_vars = ['a', 'b', 'c']
atoms = [('P', ('a', 'b')), ('Q', ('b', 'c'))]

# [(2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4)]
# [(3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5), (5, 5)]
a_list =   [2, 2, 3, 3, 3, 4, 4]
b_list_1 = [3, 4, 2, 3, 4, 3, 4]
b_list_2 = [3, 3, 3, 4, 4, 4, 5]
c_list =   [3, 4, 5, 3, 4, 5, 5]

data = {
    'P': {
        'a': a_list,
        'b': b_list_1,
    },
    'Q': {
        'b': b_list_2,
        'c': c_list,
    },
}
sum_order = {'c':-1, 'a':1, }
test_s()


# %%
# Larger stress test. Full acyclic CQ
free_vars = ['a', 'b', 'c']
atoms = [('P', ('a', 'b')), ('Q', ('b', 'c'))]

a_vals = range(1, 5)
b_vals = range(1, 5)
product = list(itertools.product(a_vals, b_vals))
a_list, b_list_1 = zip(*product)
a_list = list(a_list)
b_list_1 = list(b_list_1)

b_vals = range(2, 6)
c_vals = range(2, 6)
product = list(itertools.product(b_vals, c_vals))
b_list_2, c_list = zip(*product)
b_list_2 = list(b_list_2)
c_list = list(c_list)

data = {
    'P': {
        'a': a_list,
        'b': b_list_1,
    },
    'Q': {
        'b': b_list_2,
        'c': c_list,
    },
}
sum_order = ['a', 'b', 'c']
# sum_order = {'c':-1, 'a':1, }
test_s()
