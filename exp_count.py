import math
import json
from pathlib import Path
from select_k.Query import ConjunctiveQuery
from select_k.LayeredAlgorithm import LayeredJoinTree
from exp_utils import parse_args
from exp_timer.exp_timer import CONFIG

def gen_quartile(res_len): 
    if res_len >= 4:
        return [0, int(0.25 * res_len) - 1, int(0.50 * res_len) - 1, int(0.75 * res_len) - 1, res_len - 1] 
    elif res_len == 3:
        return [0, 0, 1, 1, 2]
    elif res_len == 2:
        return [0, 0, 0, 0, 1]
    elif res_len == 1 :
        return [0, 0, 0, 0, 0]
    else: 
        return []
    
def main(): 
    args = parse_args()
    CONFIG.__init__(enabled=args.timer_enabled)

    query_file = Path(args.query_file).resolve()
    data_dir = Path(args.data_dir).resolve() if hasattr(args, "data_dir") else query_file.parent

    cq = ConjunctiveQuery.from_query_file(query_file=query_file, data_dir=data_dir)
    tree = LayeredJoinTree(cq)
    tree.build_layered_join_tree()
    tree.direct_access_preprocessing()
    res_len = tree.direct_access_tree[1].buckets[()]['weight'] 
    max_k = res_len - 1 if res_len > 0 else None

    max_exp = int(math.log10(max_k))
    k_list = [10**i for i in range(max_exp + 1) if 10**i <= max_k]
    if max_k not in k_list:
        k_list.append(max_k) 
    k_sql_list = [10**i for i in range(0, 7) if 10**i <= max_k]
    k_sql_list_ext = [k for k in k_list if k not in k_sql_list]
    quartile_list = gen_quartile(res_len)
    median_list = [int(0.50 * res_len) - 1]

    with open(data_dir / "k_list.json", "w") as f:
        json.dump(k_list, f, indent=2)

    with open(data_dir / "k_sql_list.json", "w") as f:
        json.dump(k_sql_list, f, indent=2)
    
    with open(data_dir / "k_sql_list_ext.json", "w") as f:
        json.dump(k_sql_list_ext, f, indent=2)

    with open(data_dir / "quartile_list.json", "w") as f:
        json.dump(quartile_list, f, indent=2)

    with open(data_dir / "median_list.json", "w") as f:
        json.dump(median_list, f, indent=2)

if __name__ == "__main__":
    main()



