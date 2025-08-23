import random
from select_k.Query import ConjunctiveQuery
from select_k.JoinTreeNode import JoinTreeNode
from select_k.Selection import Selection
import numpy as np
from collections import defaultdict
from exp_timer.exp_timer import timer

class Selection_Sum: 
    @timer(name="PrepareSelection", extra=lambda ctx: f"exp={ctx.exp_id}_trial={ctx.trial}" if hasattr(ctx, 'exp_id') and hasattr(ctx, 'trial') else None)
    def __init__(self, cq: ConjunctiveQuery):
        """
        Initialize Selection Class to determine if the selection is possible (in quasilinear time).
        """
        self.query = cq
        if not self.query.free_connex: 
            raise Exception("NOT Tractable for Selection.")
        # For now, support only full queries with 2 atoms
        elif not self.query.full or len(self.query.atoms) > 2:
            raise Exception("NOT Tractable for Selection.")
        else: 
            print('Tractable for Selection.') 

    @timer(name="SelectK", extra=lambda ctx: f"exp={ctx.exp_id}_trial={ctx.trial}" if hasattr(ctx, 'exp_id') and hasattr(ctx, 'trial') else None)
    def select_k(self, k:int): 
        """
        Select the k-th record according to the sum order.
        """

        # build a join tree (in the general case, we need to know what are the 2 adjacent nodes that contain the SUM variables)
        root = self.query.build_join_tree() 
        
        # To calculate the correct SUMs we need to assign each SUM variable to a specific join tree node
        # (Otherwise, we would for example double-count b in R(a, b), S(b, c))
        # However, double-counting does not influence the relative position of the tuples
        # So, it is fine to skip that step and double-count some variables

        # Pick a good pivot (i.e., an answer that is relatively in the middle of the ranking)
        pivot = self.bottom_up_pivot(root, self.query.sum_order)
    
    @staticmethod
    def bottom_up_pivot(node:JoinTreeNode, sum_order): 
        # print('BOTTOM UP PIVOT') 
        # print(f'go into {node}')

        # Compute for each tuple:
        # 1) the count of the answers in the subtree
        node.select_count = np.ones(len(node.relation.instance_row), dtype=int)
        # 2) a pivot for its subtree as the union of its values together with the pivots of its children
        node.select_count = np.zeros(len(node.relation.instance_row), dtype=int)
        for i, tup in enumerate(node.relation.instance_row):
            # to check if this is correct
            node.pivots[i] = sum(tup[node.relation.variables.index(var)] for var in sum_order if var in node.relation.variables)



        # the pivot of a child is calulated as the weighted median of the pivots of the connecting tuples

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