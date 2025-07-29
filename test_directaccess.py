# %%
from select_k.Query import ConjunctiveQuery
from select_k.LayeredAlgorithm import LayeredJoinTree
from select_k.Relation import Relation 
import pandas as pd

# %%
def direct_access_test_k(atoms, free_vars, lex_order, data, k): 
    cq = ConjunctiveQuery(atoms, free_vars, lex_order, data=data)
    ljt = LayeredJoinTree(cq)
    ljt.build_layered_join_tree()
    ljt.direct_access_preprocessing()
    result = ljt.direct_access(k)
    print(f'[{k}]: {result}')
    return result

def direct_access_preprocess(atoms, free_vars, lex_order, data): 
    cq = ConjunctiveQuery(atoms, free_vars, lex_order, data=data)
    ljt = LayeredJoinTree(cq)
    ljt.build_layered_join_tree()
    # print(ljt)
    ljt.direct_access_preprocessing() 
    return ljt

def direct_access_result(layer_join_tree, k): 
    result = layer_join_tree.direct_access(k)
    return result

def direct_access(atoms, free_vars, lex_order, data): 
    result_list = []
    ljt = direct_access_preprocess(atoms, free_vars, lex_order, data)
    max_k = ljt.direct_access_tree[1].buckets[()]['weight']
    output = ""
    for i in range(max_k):
        result = direct_access_result(ljt, i)
        output += f"k: {i}, RESULT: {result} \n"
        result_list.append(result)

    print(output)

    return result_list
    

def smart_join_and_sort(atoms, data, lex_order, free_varaibles):
    df_result = None

    for rel_name, rel_vars in atoms: 

        if isinstance(data[rel_name], list): 
            tuples = data[rel_name]
            columns = list(zip(*tuples))
            data_dict = {var: list(col) for var, col in zip(rel_vars, columns)}

        else: 
            data_dict = data[rel_name]

        df_current = pd.DataFrame({var: data_dict[var] for var in rel_vars})

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
    return df_result[leading_cols].drop_duplicates().reset_index()

# %%
def test_d(): 
    df_result = smart_join_and_sort(atoms, data, lex_order, free_vars)
    print(df_result)
    access_list = direct_access(atoms, free_vars, lex_order, data) 
    access_lex_list = [{k: v for k, v in d.items() if k != 'index' and k in lex_order} for d in access_list]

    df_list = df_result.to_dict(orient='records') 
    df_list = [{k: v for k, v in d.items() if k != 'index'} for d in df_list]
    df_lex_list = [{k: v for k, v in d.items() if k != 'index' and k in lex_order} for d in df_list]
    # flag = 1
    # for i, (row_dict, ref_dict) in enumerate(zip(df_list, access_list)):
    #     if row_dict != ref_dict:
    #         flag = 0
    #         print(f"Mismatch at row {i}:\n  df:  {row_dict}\n  ref: {ref_dict}")
    # if flag: 
    #     print("PASS ALL")

    flag2 = 1
    for i, (row_dict, ref_dict) in enumerate(zip(df_lex_list, access_lex_list)):
        if row_dict != ref_dict:
            flag2 = 0
            print(f"Mismatch at row {i}:\n  df:  {row_dict}\n  ref: {ref_dict}")
    if flag2: 
        print("PASS ALL")


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
# lex_order = ['a', 'b', 'c']
lex_order = {'a':-1, 'b':1, }
test_d()

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
test_d()

# %%
"""
Test for single k
"""
# k = 2
# result = direct_access_test_k(atoms, free_vars, lex_order, data, k)
# %%
