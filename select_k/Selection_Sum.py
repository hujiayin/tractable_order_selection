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
        # if not self.query.free_connex: 
        #     raise Exception("NOT Tractable for Selection.")
        # # For now, support only full queries with 2 atoms
        # elif not self.query.full or len(self.query.atoms) > 2:
        #     raise Exception("NOT Tractable for Selection.")
        # else: 
        #     print('Tractable for Selection.') 

    @timer(name="SelectK", extra=lambda ctx: f"exp={ctx.exp_id}_trial={ctx.trial}" if hasattr(ctx, 'exp_id') and hasattr(ctx, 'trial') else None)
    def select_k(self, k:int): 
        """
        Select the k-th record according to the sum order.
        """

        # build a join tree (in the general case, we need to know what are the 2 adjacent nodes that contain the SUM variables)
        root = self.query.build_join_tree_arbitrary_root() 
        
        # To calculate the correct SUMs we need to assign each SUM variable to a specific join tree node
        # (Otherwise, we would for example double-count b in R(a, b), S(b, c))
        # However, double-counting does not influence the relative position of the tuples
        # So, it is fine to skip that step and double-count some variables

        # Pick a good pivot (i.e., an answer that is relatively in the middle of the ranking)
        pivot = self.pick_pivot(self.query, root)

        return pivot

    @staticmethod
    def pick_pivot(query, root):

        def bottom_up_pivot(node:JoinTreeNode, sum_order): 
            # print('BOTTOM UP PIVOT') 
            # print(f'go into {node}')

            # Compute for each tuple:
            # 1) the count of the answers in the subtree
            node.select_count = np.ones(len(node.relation.instance_row), dtype=int)
            # 2) a pivot for its subtree as the union of its values together with the pivots of its children
            # a pivot is represented as a dict from variable names to their values (all variables, not just those in the sum)
            node.pivots = np.empty(len(node.relation.instance_row), dtype=object)
            for i in range(len(node.pivots)):
                node.pivots[i] = {}
            # For all the tuples of the relation, we need to calculate the SUM
            # For that, we need the indexes of the SUM variables in the current relation, together with the factors we multiply them
            sum_indexes, factors = zip(*[(node.relation.variables.index(var), weight) for var, weight in sum_order.items() if var in node.relation.variables])
            for i, tup in enumerate(node.relation.instance_row):
                for j, var in enumerate(node.relation.variables):
                    node.pivots[i][var] = tup[j]
                # Calculate the SUM value of the tuple, taking into account the constant factors stored in the sum_order dict
                node.pivots[i]['sum'] = sum(tup[j] * weight for j, weight in zip(sum_indexes, factors))

            if node.children:
                for child, conn in node.children_connection.items(): 
                    # print(child, conn)
                    bottom_up_pivot(child, sum_order) 
                    
                    # the pivot of a child is calulated as the weighted median of the pivots of the connecting tuples

                    if not conn:
                        # no connections
                        # calculate the counts
                        total_weight = np.sum(child.select_count)
                        node.select_count *= total_weight
                        # calculate the weighted median of the child
                        child_pivot = weighted_median_linear(child.pivots, child.select_count, total_weight)
                        node.pivots.update(child_pivot)
                        # print(node.select_count)
                        # print(node.pivots)
                        continue

                    parent_key_idx = [node.relation.variables.index(attr) for attr in conn]
                    child_key_idx = [child.relation.variables.index(attr) for attr in conn]

                    # Each bucket has the pivots that share the same join key together with their weights
                    child_buckets = defaultdict(list)
                    for tup, w, p in zip(child.relation.instance_row, child.select_count, child.pivots):
                        key = tuple(tup[i] for i in child_key_idx)
                        child_buckets.setdefault(key, []).append((p, w))
                    # The messages sent from a child contain a tuple of aggregates for each join key (weighted median for pivots, sum for counts)
                    child_messages = {}
                    for key, bucket in child_buckets.items():
                        child_pivots, child_weights = zip(*bucket)
                        total_bucket_weight = sum(child_weights)
                        child_messages[key] = (weighted_median_linear(child_pivots, child_weights, total_bucket_weight), total_bucket_weight)

                    for i, (tup, w, p) in enumerate(zip(node.relation.instance_row, node.select_count, node.pivots)):
                        key = tuple(tup[i] for i in parent_key_idx)
                        (child_pivot, child_weight) = child_messages.get(key, (None, 0))
                        ## TODO: Maybe that's the place we should delete a tuple if it doesn't join (the weight map returns 0)
                        node.select_count[i] = w * child_weight
                        if child_pivot is not None:
                            node.pivots[i].update(child_pivot)
            # print(node.relation.instance_row)
            # print(node.select_count)
            # print(f'go out {node}')

        # Compute a pivot for each tuple (corresponding to its joining subtree)
        bottom_up_pivot(root, query.sum_order)
        # Take the weighted median of the pivots of the root
        final_pivot = weighted_median_linear(root.pivots, root.select_count, sum(root.select_count))
        return final_pivot

        



def weighted_median_linear(elements, weights, total_weight):
    """
    Compute the weighted median in expected linear time using Quickselect.
    elements: each element is a dict that contains a key 'sum'. This is what we compare with.
    weights: the number of times each element appears, same length as values
    total_weight: float, the total weight of all tuples
    """

    elements = np.array(elements, dtype=object)  # Enable boolean indexing
    print(elements)
    weights = np.asarray(weights)
    assert len(elements) == len(weights)

    if len(elements) == 1:
        return elements[0]
    pivot_idx = np.random.randint(len(elements))
    pivot = elements[pivot_idx]
    left_mask = np.array([t['sum'] < pivot['sum'] for t in elements])
    right_mask = np.array([t['sum'] > pivot['sum'] for t in elements])
    eq_mask = np.array([t['sum'] == pivot['sum'] for t in elements])

    w_left = weights[left_mask].sum()
    w_eq = weights[eq_mask].sum()
    if w_left > total_weight / 2:
        return weighted_median_linear(elements[left_mask], weights[left_mask], total_weight)
    elif w_left + w_eq < total_weight / 2:
        return weighted_median_linear(elements[right_mask], weights[right_mask], total_weight)
    else:
        return pivot

