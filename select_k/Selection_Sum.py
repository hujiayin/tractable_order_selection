from logging import root
from platform import node
import random
import copy
from select_k.Query import ConjunctiveQuery
from select_k.JoinTreeNode import JoinTreeNode
from select_k.Selection import Selection
from select_k.Relation import Relation
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

        # build a join tree (in the general case, we need to know what are the 2 adjacent nodes that contain the SUM variables)
        self.root = self.query.build_join_tree_arbitrary_root() 

        # assume that the join tree has been oriented in a way such that the SUM variables are contained in the root and its first child
        # self.node_with_sum_1 = self.root
        # self.node_with_sum_2 = self.root.children[0]

        # assign each SUM variable to one of the 2 relations
        # sum_vars_1 and sum_vars_2 are lists that store (index of sum var, factor) for each of the 2 relations
        # sum_vars_1 refers to the root and sum_vars_2 refers to its first child
        # for sum_vars_2, flip the sign of the factors so that we can have the variables on each side of an inequality such as a + b < -c + pivot
        self.sum_vars_1 = []
        self.sum_vars_2 = []
        for var in self.query.sum_order:
            if var in self.root.relation.variables:
                self.sum_vars_1.append((self.root.relation.variables.index(var), self.query.sum_order[var]))
            elif var in self.root.children[0].relation.variables:
                self.sum_vars_2.append((self.root.children[0].relation.variables.index(var), -self.query.sum_order[var]))
            else:
                raise Exception(f"SUM variable {var} not found in either of the 2 designated relations.")

    @timer(name="SelectK", extra=lambda ctx: f"exp={ctx.exp_id}_trial={ctx.trial}" if hasattr(ctx, 'exp_id') and hasattr(ctx, 'trial') else None)
    def select_k(self, k:int): 
        """
        Select the k-th record according to the sum order.
        """

        # To calculate the correct SUMs we need to assign each SUM variable to a specific join tree node
        # (Otherwise, we would for example double-count b in R(a, b), S(b, c))
        # However, double-counting does not influence the relative position of the tuples
        # So, it is fine to skip that step and double-count some variables

        current_root = self.root

        while True:
            # Pick a good pivot (i.e., an answer that is relatively in the middle of the ranking)
            pivot = self.pick_pivot(current_root, self.query.sum_order)
            pivot_sum = sum(pivot[var] for var in self.query.sum_order)
            pivot['sum'] = pivot_sum

            remaining_answers = sum(current_root.select_count)

            # Create a new query (with a new database) that produces the answers that are smaller than the pivot
            # (satisfying the inequality SUM < pivot['sum'])
            # For simplicity, we directly work on the join tree, not the query structure
            root_less_than = self.trim_lt_inequality(current_root, self.sum_vars_1, self.sum_vars_2, pivot_sum)
            count_less_than = Selection.bottom_up_count(root_less_than)

            root_greater_than = self.trim_gt_inequality(current_root, self.sum_vars_1, self.sum_vars_2, pivot_sum)
            count_greater_than = Selection.bottom_up_count(root_greater_than)

            count_equal = remaining_answers - count_less_than - count_greater_than

            if k < count_less_than:
                current_root = root_less_than
                remaining_answers = count_less_than
            elif k < count_less_than + count_equal:
                # We found the k-th answer
                # Return it after removing the auxiliary variables
                if '__lt_var' in pivot:
                    del pivot['__lt_var']
                if '__gt_var' in pivot:
                    del pivot['__gt_var']
                return pivot
            else:
                current_root = root_greater_than
                remaining_answers = count_greater_than
                k = k - count_less_than - count_equal

            # TODO: the iterative process should stop early, when the remaining answers are significantly less than the database size
            # At that point, we should just join and sort

    @staticmethod
    def trim_lt_inequality(root: JoinTreeNode, sum_vars_root: list, sum_vars_child: list, offset: float):
        Selection_Sum.trim_inequality(root, sum_vars_root, sum_vars_child, offset, -1)

    @staticmethod
    def trim_gt_inequality(root: JoinTreeNode, sum_vars_root: list, sum_vars_child: list, offset: float):
        Selection_Sum.trim_inequality(root, sum_vars_root, sum_vars_child, offset, 1)

    @staticmethod
    def trim_inequality(root: JoinTreeNode, sum_vars_root: list, sum_vars_child: list, offset: float, direction: int):
        """
        Assuming that the root and its first child contain all SUM variables, enforce the inequality SUM_root < -SUM_child + offset or SUM_root > -SUM_child + offset
        Returns a new join tree with modified relations (without destroying the original one)
        root: the root of the join tree
        sum_vars_root: a list of (index, factor) for the root variables
        sum_vars_child: a list of (index, factor) for the child variables, where the factors have been negated
        offset: float, the offset to apply
        direction: int, the direction of the inequality (1 for >, -1 for <)
        """

        # for key in root.children_connection.keys():
        #     if key is not root.children[0]:
        #         print("NOT THE SAME")
        #         print(key)
        #         print(root.children[0])
        #     else:
        #         print("SAME")

        # Initialize the new relations
        new_relation_root = copy.copy(root.relation)
        new_relation_root.instance_row = []
        new_relation_root.rowid = None
        new_relation_child = copy.copy(root.children[0].relation)
        new_relation_child.instance_row = []
        new_relation_child.rowid = None

        # Clone the join tree, with the same relation pointers
        # Since we will only modify the root node and its first child node, we can clone only those
        new_root = copy.copy(root)
        new_root.select_count = None
        new_root.pivots = None
        new_root.children = copy.copy(root.children)
        new_root.children_connection = copy.copy(root.children_connection)
        new_root.children[0] = copy.copy(root.children[0])
        new_root.children[0].select_count = None
        new_root.children[0].pivots = None
        new_root.children[0].parent_connection = copy.copy(root.children[0].parent_connection)

        # children_connection is a dict with join tree nodes as keys so we need to modify it now that we changed the child node
        del new_root.children_connection[root.children[0]]
        new_root.children_connection[new_root.children[0]] = root.children_connection[root.children[0]]

        new_root.relation = new_relation_root
        new_root.children[0].relation = new_relation_child


        # Create new auxiliary variables between the two relations that will encode the inequality condition
        # (If such a variable already exists, we reuse it)
        new_var_name = '__lt_var' if direction == -1 else '__gt_var'
        # root
        if new_var_name in new_relation_root.variables:
            ineq_var_root_idx = new_relation_root.variables.index(new_var_name)
        else:
            new_relation_root.variables = new_relation_root.variables + (new_var_name,)
            ineq_var_root_idx = len(new_relation_root.variables) - 1
            # Adjust the connection of the root to the child, adding the new variable to the existing connection
            new_root.children_connection[new_root.children[0]].add(new_var_name)
        # child
        if new_var_name in new_relation_child.variables:
            ineq_var_child_idx = new_relation_child.variables.index(new_var_name)
        else:
            new_relation_child.variables = new_relation_child.variables + (new_var_name,)
            ineq_var_child_idx = len(new_relation_child.variables) - 1
            # Adjust the parent connection of the child to the root, adding the new variable to the existing connection
            new_root.children[0].parent_connection.add(new_var_name)

        # Now we are ready to modify the data
        Selection_Sum.trim_data_inequality(new_relation_root, new_relation_child, root.relation, root.children[0].relation, root.children_connection[root.children[0]],
                                                ineq_var_root_idx, ineq_var_child_idx, sum_vars_root, sum_vars_child, offset, direction)

        return new_root
    
    @staticmethod
    def trim_data_inequality(new_parent: Relation, new_child: Relation, parent: Relation, child: Relation, connection: list, \
                                ineq_var_parent_idx: int, ineq_var_child_idx: int, \
                                sum_vars_parent: list, sum_vars_child: list, offset: float, direction: int):
        """
        Enforce the inequality SUM_parent < -SUM_child + offset on the given relations.
        Populates new_parent and new_child with data.
        """
        new_var_value_id = 0

        def connect_tuples(parent_tuples, child_tuples):
            # Connects these groups of tuples by associating them with the same value for the inequality variable
            nonlocal new_var_value_id
            new_var_value_id += 1
            for parent_row in parent_tuples:
                # Set a common value for the inequality variable in this group of tuples, and remove the sum value at the end
                new_parent_row = parent_row[:ineq_var_parent_idx] + (new_var_value_id,) + parent_row[(ineq_var_parent_idx + 1):-1]
                new_parent.instance_row.append(new_parent_row)
            for child_row in child_tuples:
                new_child_row = child_row[:ineq_var_child_idx] + (-new_var_value_id,) + child_row[(ineq_var_child_idx + 1):-1]
                new_child.instance_row.append(new_child_row)


        def recursive_partitioning(parent_tuples_sorted, child_tuples_sorted, distinct_vals_sorted):
            num_distinct_vals = len(distinct_vals_sorted)
            # Base case
            if num_distinct_vals <= 1:
                # Inequality cannot be satisfied
                return
            # Recursive case
            # Split the distinct sums in half
            mid_distinct = num_distinct_vals // 2
            low_distinct_vals = distinct_vals_sorted[:mid_distinct]
            high_distinct_vals = distinct_vals_sorted[mid_distinct:]
            breakpoint_val = distinct_vals_sorted[mid_distinct]

            # Partition the tuples on low/high
            parent_breakpoint = len(parent_tuples_sorted)
            for (i, tup) in enumerate(parent_tuples_sorted):
                if tup[-1] >= breakpoint_val:
                    parent_breakpoint = i
            child_breakpoint = len(child_tuples_sorted)
            for (i, tup) in enumerate(child_tuples_sorted):
                if tup[-1] >= breakpoint_val:
                    child_breakpoint = i
            parent_low = parent_tuples_sorted[:parent_breakpoint]
            parent_high = parent_tuples_sorted[parent_breakpoint:]
            child_low = child_tuples_sorted[:child_breakpoint]
            child_high = child_tuples_sorted[child_breakpoint:]

            if direction == -1:
                # Less than: Connect parent_low to child_high
                if len(parent_low) > 0 and len(child_high) > 0:
                    connect_tuples(parent_low, child_high)

                # Continue recursively in each low/high partition
                if len(parent_low) > 0 and len(child_low) > 0:
                    recursive_partitioning(parent_low, child_low, low_distinct_vals)
                if len(parent_high) > 0 and len(child_high) > 0:
                    recursive_partitioning(parent_high, child_high, high_distinct_vals)
            else:
                # Greater than: Connect parent_high to child_low
                if len(parent_high) > 0 and len(child_low) > 0:
                    connect_tuples(parent_high, child_low)

                # Continue recursively in each low/high partition
                if len(parent_low) > 0 and len(child_low) > 0:
                    recursive_partitioning(parent_low, child_low, low_distinct_vals)
                if len(parent_high) > 0 and len(child_high) > 0:
                    recursive_partitioning(parent_high, child_high, high_distinct_vals)

        # End of internal functions
        ################################################
        ################################################

        # First partition the rows of parent-child based on the equality condition, which is stored in the connection between them
        parent_partition = defaultdict(list)
        child_partition = defaultdict(list)

        # If the inequality variable was already in the connection, remove it when checking for equality
        # The reason is that we want to overwrite the variable with new values
        modified_connection = connection.copy()
        new_var_name = '__lt_var' if direction == -1 else '__gt_var'
        if new_var_name in modified_connection:
            modified_connection.remove(new_var_name)
            new_var_already_exists = True
        parent_key_idx = [parent.variables.index(attr) for attr in modified_connection]
        child_key_idx = [child.variables.index(attr) for attr in modified_connection]

        # TODO: handle the case of empty connection

        # Create copies of the tuples that contain 
        # 1) the new variable (if they don't) at the correct index and its value is 0
        # 2) the SUM value as the last element (this includes the offset for the child)
        # Then, partition the rows so that each partition has the rows that share the same join key
        for row in parent.instance_row:
            sum_val = sum(row[indx] * factor for (indx, factor) in sum_vars_parent)
            if not new_var_already_exists:
                new_row = row + (0, sum_val)  # add the new variable initialized to 0
            else:
                new_row = row[:ineq_var_parent_idx] + (0,) + row[ineq_var_parent_idx:] + (sum_val,)
            key = tuple(row[i] for i in parent_key_idx)
            parent_partition.setdefault(key, []).append(new_row)
        for row in child.instance_row:
            sum_val = sum(row[indx] * factor for (indx, factor) in sum_vars_child) + offset
            if not new_var_already_exists:
                new_row = row + (0, sum_val)
            else:
                new_row = row[:ineq_var_child_idx] + (0,) + row[ineq_var_child_idx:] + (sum_val,)
            key = tuple(row[i] for i in child_key_idx)
            child_partition.setdefault(key, []).append(new_row)

        for key in parent_partition.keys():
            if not child_partition[key]:
                continue
            parent_tuples = parent_partition[key]
            child_tuples = child_partition[key]

            # If the inequality variable already existed, we can have duplicate tuples
            if new_var_already_exists:
                parent_tuples = list(set(parent_tuples))
                child_tuples = list(set(child_tuples))

            # Sort each tuple set based on the sum value (the last element of each tuple)
            # This sorting is maintained across all recursive calls
            parent_tuples.sort(key=lambda x: x[-1])
            child_tuples.sort(key=lambda x: x[-1])

            # Put the distinct sum values in 1 common list
            distinct_sums = set()
            for tup in parent_tuples:
                distinct_sums.add(tup[-1])
            for tup in child_tuples:
                distinct_sums.add(tup[-1])
            distinct_sums_sorted = sorted(distinct_sums)

            recursive_partitioning(parent_tuples, child_tuples, distinct_sums_sorted)

            # TODO: Not sure if this is necessary
            new_parent.rowid = list(range(len(new_parent.instance_row)))
            new_child.rowid = list(range(len(new_child.instance_row)))

    @staticmethod
    def pick_pivot(root, sum_order):

        def bottom_up_pivot(node:JoinTreeNode): 
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
                    bottom_up_pivot(child) 
                    
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
        bottom_up_pivot(root)
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

