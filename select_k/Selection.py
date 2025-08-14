import random
from select_k.Query import ConjunctiveQuery
from select_k.JoinTreeNode import JoinTreeNode
import numpy as np
from collections import defaultdict
from exp_timer.exp_timer import timer

class Selection: 
    @timer(name="PrepareSelection", extra=lambda ctx: f"exp={ctx.exp_id}_trial={ctx.trial}" if hasattr(ctx, 'exp_id') and hasattr(ctx, 'trial') else None)
    def __init__(self, cq: ConjunctiveQuery):
        """
        Initialize Selection Class to determine if the selection is possible.
        """
        self.query = cq
        if not self.query.free_connex: 
            raise Exception("NOT Tractable for Selection.")
        else: 
            print('Tractable for Selection.') 

    @timer(name="SelectK", extra=lambda ctx: f"exp={ctx.exp_id}_trial={ctx.trial}" if hasattr(ctx, 'exp_id') and hasattr(ctx, 'trial') else None)
    def select_k(self, k:int): 
        """
        Select the k-th record according to the lexicographic order.
        """
        cur_k = k 
        result_record = {}  # {variable: value}

        if not self.query.full: 
            self.query.build_auxiliary_tree(is_selection=True) 
        
        # lex order 
        if self.query.partial_lex: 
            self.query.lex_order_plus_greedy()
            lex_order = self.query.lex_order_plus
        else: 
            lex_order = self.query.lex_order

        # original join tree (the first root contains the first variable in lex order)
        root = self.query.build_join_tree() 


        # build a tree for each variable in lex_order 
        for i, lex_variable in enumerate(lex_order): 

            # i = 0, the root contains the first var in LEX
            if i >= 1:
                # change the root of the join tree 
                root = JoinTreeNode.find_node_with_var_bfs(root, lex_variable) 
                JoinTreeNode.reroot_tree(root) 
            
            if self.query.partial_lex and result_record:
                # when it is partial lex order, for the variable whose value has been determined
                # in the previous loops, filter the data to keep those with the determined tuples 
                variables = root.relation.variables
                relevant_indices = [i for i, var in enumerate(variables) if var in result_record]
                relevant_values = [result_record[variables[i]] for i in relevant_indices]
                root.relation.instance_row = [
                    row for row in root.relation.instance_row
                    if all(row[i] == relevant_values[j] for j, i in enumerate(relevant_indices))
                ]

            var_index = root.relation.variables.index(lex_variable)

            # bottom-up count
            Selection.bottom_up_count(root) 

            if self.query.lex_dict[lex_variable] == 0: # no order 
                var_vals = [row[var_index] for row in root.relation.instance_row]
                # print(f'current_k: {cur_k}, Values: {var_vals}, {root.select_count}')
                total = 0
                for j, ct in enumerate(root.select_count): 
                    total += ct
                    if cur_k < total: 
                        result_value = var_vals[j]
                        cur_k -= sum(x for x in root.select_count[:j])
                        # result_record[lex_variable] = result_value 
                        break
                else: 
                    raise IndexError("k exceeds total weight")
                
            else:  
                if_reverse = False if self.query.lex_dict[lex_variable] == 1 else True

                # value of lex_variable 
                weights = defaultdict(int)
                for j, row in enumerate(root.relation.instance_row): 
                    weights[row[var_index]] += root.select_count[j]
                weights = {k: v for k, v in weights.items() if v > 0}

                # quick selection to get the value for cur_k
                result_value, w_L = Selection.quick_select(weights, k=cur_k, reverse=if_reverse)
                # calculate remaining k to update k for the next variable
                cur_k -= w_L 

            # add to the result record
            result_record[lex_variable] = result_value 
            # filter data 
            root.relation.instance_row = [
                row for row in root.relation.instance_row if row[var_index] == result_value
            ]

        # print(f'------------{result_record}')
        return result_record
    
    @staticmethod
    def quick_select(dict_weights, k, reverse=False):
        if k < 0:
            raise ValueError("k should be non-negative")

        items = list(dict_weights.items())

        def quick_select_weighted(items, k, pre_weight): 
            # print(items, k)
            if not items:
                raise IndexError("k exceeds total weight")

            pivot_key, _ = random.choice(items)

            def compare(x):
                return x[0] > pivot_key if reverse else x[0] < pivot_key

            L = [x for x in items if compare(x)] # left part
            E = [x for x in items if x[0] == pivot_key]
            R = [x for x in items if not compare(x) and x[0] != pivot_key]

            w_L = sum(w for _, w in L)
            w_E = sum(w for _, w in E)

            if k < w_L: 
                return quick_select_weighted(L, k, pre_weight)
            elif k < w_L + w_E: 
                return pivot_key, pre_weight + w_L
            else: 
                return quick_select_weighted(R, k - w_L - w_E, pre_weight + w_L + w_E)

        return quick_select_weighted(items, k, 0)
    
    @staticmethod
    def bottom_up_count(node:JoinTreeNode): 
        # print('BOTTOM UP') 
        # print(f'go into {node}')

        node.select_count = np.ones(len(node.relation.instance_row), dtype=int)

        if node.children:
            for child, conn in node.children_connection.items(): 
                # print(child, conn)
                Selection.bottom_up_count(child) 
                
                if not conn:
                    # no connections
                    total_weight = np.sum(child.select_count)
                    node.select_count *= total_weight 
                    # print(node.select_count)
                    continue

                parent_key_idx = [node.relation.variables.index(attr) for attr in conn]
                child_key_idx = [child.relation.variables.index(attr) for attr in conn]

                weight_map = defaultdict(int) 

                # weight_map in children
                for tup, w in zip(child.relation.instance_row, child.select_count):
                    key = tuple(tup[i] for i in child_key_idx)
                    weight_map[key] += w

                new_weights = []
                for tup, w in zip(node.relation.instance_row, node.select_count):
                    key = tuple(tup[i] for i in parent_key_idx)
                    new_weight = w * weight_map.get(key, 0)
                    new_weights.append(new_weight)
                node.select_count = np.array(new_weights) 
        # print(node.relation.instance_row)
        # print(node.select_count)
        # print(f'go out {node}')