from itertools import combinations
from select_k.Query import ConjunctiveQuery
from select_k.JoinTreeNode import JoinTreeNode
from select_k.Relation import Relation
import numpy as np
from collections import defaultdict
from exp_timer.exp_timer import timer

class LayeredJoinTree:
    @timer(name="PrepareTree", extra=lambda ctx: f"exp={ctx.exp_id}_trial={ctx.trial}" if hasattr(ctx, 'exp_id') and hasattr(ctx, 'trial') else None)
    def __init__(self, cq: ConjunctiveQuery):
        """
        Initialize the Join Tree constructor.
        """
        self.query = cq
        
        self.lex_pos = {var: idx + 1 for idx, var in enumerate(cq.lex_order)} # position of variables in lex order 

        self.neighbor_map = self.build_neighbor_map() 

        # Raise error if the cq does not satisfy the requirements to build a join tree
        if not self.ispossible_tree(): 
            raise Exception("Cannot build a Layered Join Tree.")
        else: 
            print('possible to build a layered join tree')

        self.direct_access_tree = {}
            
    def __repr__(self):
        """
        Show the Layered Join Tree.
        """
        lines = []
        for layer, node in self.direct_access_tree.items():
            lines.append(f"Layer {layer}: \n{node}\n   Children → {node.children}\n   Parent → {node.parent} \
                         \n   Relation:{node.relation}")
        return "\n".join(lines)
    
    # build neighbor map for each variable
    def build_neighbor_map(self):
        neighbor_map = {v: set() for v in self.query.vars}
        for atom in self.query.atoms: 
            for v1 in atom.variables: 
                for v2 in atom.variables:
                    if v1 != v2:
                        neighbor_map[v1].add(v2)
        return neighbor_map

    @classmethod
    def find_disruptive_trios(cls, neighbor_map, lex_pos):
        """
        Find all disruptive trios given a neighbor map and a partial lexicographic order.

        :param neighbor_map: dict[var] -> set of neighboring vars
        :param lex_order: list of variables (partial or full order)
        :return: list of disruptive trios (tuples of form (a, b, c))
        """
        if len(lex_pos) < 3:
            return []
        
        result = []
        sorted_lex = sorted(lex_pos.keys(), key=lambda k: lex_pos[k])
       
        for x, y, z in list(combinations(sorted_lex, 3)):
            if y not in neighbor_map.get(x, set()) and \
            z in neighbor_map.get(x, set()) and \
            z in neighbor_map.get(y, set()) and \
            lex_pos[z] > lex_pos[x] and \
            lex_pos[z] > lex_pos[y]:
                result.append((x, y, z))
        print('disruptive trios detected:', result)
        return result

    def detect_disruptive_trios(self): 
        if LayeredJoinTree.find_disruptive_trios(self.neighbor_map, self.lex_pos) == []: 
            return False
        return True
    
    # determine if the cq is possible for a join tree
    def ispossible_tree(self): 
        # lex order
        if self.query.lex_order: # not None. ie lex order 
            if self.query.free_connex == False: 
                print('The query is not free-connex.')
                return False
            elif (self.query.partial_lex == True) and (self.query.lex_connex == False): 
                print('The query is not L-connex.')
                return False
            elif self.detect_disruptive_trios(): 
                print('The query contains disruptive trios.')
                return False
            
        return True

    @timer(name="BuildTree", extra=lambda ctx: f"exp={ctx.exp_id}_trial={ctx.trial}" if hasattr(ctx, 'exp_id') and hasattr(ctx, 'trial') else None)
    def build_layered_join_tree(self):

        if not self.query.full: 
            
            self.query.build_auxiliary_tree()
        
        atoms = self.query.atoms_free

        if self.query.partial_lex: 
            
            self.query.lex_order_plus_greedy()
            lex_order = self.query.lex_order_plus
        else: 
            lex_order = self.query.lex_order

        
        for i, max_variable in enumerate(lex_order):
            # Find the max relation with projection of the first i elements in lexicographic_order
            max_relation_proj = None 
            width = 0
            current_layer = i + 1
            node_rel = None
            # find the relation or the projection of the relation assigning to layer i+1 (start from layer 1)
            for relation in atoms: 
                if max_variable in relation.variables: 
                    # intersection of the original relation and the current layer LEX order. in the order of LEX
                    variables = [v for v in lex_order[: current_layer] if v in relation.variables]
                    if len(variables) > width: 
                        width = len(variables)
                        if_projection = False
                        if len(variables) != relation.width: # not original relation. need projection
                            if_projection = True 
                        node_variable = variables
                        node_rel = relation # In the layered join tree, the order of variables in relations aligns with the lexicographic order
            
            # Assign relation to the specific node(layer)
            if if_projection: # create new relation by the projection
                name = node_rel.name + '-' + str(current_layer)
                data = Relation.project_remove_duplicates(data=node_rel.instance_row, 
                                                        orig_vars=node_rel.variables, 
                                                        new_vars=node_variable)
                max_relation_proj = Relation(name, node_variable, 
                                            instance=data, 
                                            lex_order=self.query.lex_dict, 
                                            need_check=False)
            else: 
                max_relation_proj = node_rel
                # max_relation_proj.variables = tuple(node_variable)

            # rank data in max_relation_proj
            max_relation_proj.lex_sort(self.query.lex_dict)
            
            # Add the maximal hyperedge to the tree
            treenode = JoinTreeNode(max_relation_proj, current_layer) 
            self.direct_access_tree[current_layer] = treenode 

            if current_layer == 1: 
                self.direct_access_tree_root = treenode

            # Connect to the previous layer (if exists)
            if current_layer > 1: 
                j = i
                while j >= 1: 
                    # print('Tree Layer', j, self.direct_access_tree[j])
                    if frozenset(self.direct_access_tree[j].relation.variables).issuperset(frozenset(max_relation_proj.variables) - {max_variable}): 
                        connection = frozenset(self.direct_access_tree[j].relation.variables).intersection(frozenset(max_relation_proj.variables) - {max_variable})
                        connection = [v for v in lex_order if v in connection]
                        self.direct_access_tree[j].children.append(treenode)
                        self.direct_access_tree[j].children_connection[treenode] = connection
                        self.direct_access_tree[current_layer].parent = self.direct_access_tree[j] 
                        self.direct_access_tree[current_layer].parent_connection = connection
                        break
                    j -= 1 
                

    def get_leaf_to_root_order(self):
        """
        Generate an order from leaf to root in a tree structure.

        
            tree (dict): A dictionary representing the tree structure, where the key is the layer
                        and the value is a tuple (Relation, [children_layer]).

        Returns A list of layers ordered from leaf to root.
        """
        visited = set()
        order = []

        def post_order_traversal(layer):
            if layer in visited:
                return
            visited.add(layer)
            # Get the children of the current layer
            children = self.direct_access_tree[layer].children
            # Recursively visit all children
            for child in children:
                post_order_traversal(child.layer)
            # Add the current layer after visiting its children
            order.append((layer, self.direct_access_tree[layer].parent_connection, self.direct_access_tree[layer].parent))

        # Start the traversal from all nodes (in case the tree has multiple roots)
        for layer in self.direct_access_tree:
            post_order_traversal(layer)

        return order
    
    @timer(name="PreprocessBuckets", extra=lambda ctx: f"exp={ctx.exp_id}_trial={ctx.trial}" if hasattr(ctx, 'exp_id') and hasattr(ctx, 'trial') else None)
    def direct_access_preprocessing(self):
        leaf_to_root_order = self.get_leaf_to_root_order()

        for order in leaf_to_root_order:        
            layer = order[0] 

            self.direct_access_tree[layer].preprocess_buckets()

        # for i in range(len(leaf_to_root_order)): 
        #     print(self.direct_access_tree[i+1].buckets)

    @timer(name="DirectAccess", extra=lambda ctx: f"exp={ctx.exp_id}_trial={ctx.trial}_k={ctx.k}" if hasattr(ctx, 'exp_id') and hasattr(ctx, 'trial') and hasattr(ctx, 'k') else None)
    def direct_access(self, k:int):
        """
        Access the k-th tuple in lexicographic order, using precomputed weights and indices.

        Args:
            k: int, target position (0-based)

        Returns:
            result: dict of {column: value}
        """
        # print("DIRECT ACCESS")
        tree = self.direct_access_tree
        current_k = k

        root_weight = tree[1].buckets[()]['weight']

        if k >= root_weight or k < 0: 
            raise IndexError('Out of bound')
        
        buckets_dict = {}
        buckets_dict[1] = tree[1].buckets[()]
        factor = root_weight
        result = {}

        for layer, current_node in tree.items(): 
            current_bucket = buckets_dict[layer]
            # print(current_k, current_bucket)
            factor = factor / current_bucket['weight'] 

            variables = current_node.relation.variables

            # binary search to find the bucket that k should contain in. t.start_index <= k < t.end_index
            data = current_bucket['data']
            left, right = 0, len(data) - 1
            chosen = None
            while left <= right:
                mid = (left + right) // 2
                _, _, s, e = data[mid]
                if s * factor <= current_k < e * factor:
                    chosen = data[mid]
                    break
                elif current_k < s * factor:
                    right = mid - 1
                else:
                    left = mid + 1

            if chosen is None:
                raise RuntimeError("Binary search failed.")

            row_id, weight, start, end = chosen
            current_k -= start * factor

            # get value 
            data = current_node.relation.instance_row
            for attr in variables:
                result[attr] = data[row_id][variables.index(attr)] 

            # construct the prefix for next layer
            for child, connection in current_node.children_connection.items(): 
                next_prefix = tuple(result[attr] for attr in connection)
                buckets_dict[child.layer] = child.buckets[next_prefix]
                factor = factor * child.buckets[next_prefix]['weight']

        return result
    