from copy import deepcopy
from select_k.JoinTreeNode import JoinTreeNode
from collections import defaultdict, deque
from select_k.Relation import Relation
from typing import Dict, Tuple, Optional, List, Set

class ConjunctiveQuery: 
    def __init__(self, 
                 atoms: List, 
                 free_vars: List, 
                 lex_order: List|Dict[str, int]=None, 
                 data: Optional[Dict[str, List]]=None):
        """
        Initialize a Conjunctive Query. 
        atoms: [(relation_name, (variables of the relation))]
        """
        # atoms to relations
        self.atoms = [Relation(name=atom[0], variables=atom[1], instance=data[atom[0]], lex_order=lex_order) for atom in atoms]

        self.hyperedges = [set(atom[1]) for atom in atoms]

        # var(Q): the set of variables that appear in query Q
        self.vars = {v for atom in atoms for v in atom[1]} 
        # free variables
        self.free_vars = free_vars
        self.full = True if set(self.vars) == set(self.free_vars) else False 

        # lexicographic order properties
        self.lex_order = None 
        self.lex_dict = None

        if lex_order:
            if isinstance(lex_order, List):  # ascending(1) by default
                self.lex_order = lex_order
                self.lex_dict = {v: 1 for v in lex_order} 
            elif isinstance(lex_order, Dict): # by the given ranking type [ascending(1), descending(-1)]
                self.lex_order = list(lex_order.keys())
                self.lex_dict = lex_order

        #lex order + (with all free variables)
        self.lex_order_plus = None
        if self.lex_order: 
            if set(self.lex_order) == set(self.free_vars):
                self.partial_lex = False
            else: 
                self.partial_lex = True
        
        # connexity
        self.free_connex = True if self.full else \
            self.is_x_connex_cq(self.hyperedges, self.free_vars)
        self.lex_connex = None
        if self.lex_order: 
            if self.partial_lex == False: 
                self.lex_connex = self.free_connex
            else: 
                self.lex_connex = self.is_x_connex_cq(self.hyperedges, self.lex_order)

        # for self.full=False
        self.atoms_free = self.atoms
        self.tree_aux = None

        print(f"Full CQ: {"Y" if self.full else "N"}\
              \nComplete LEX: {"Y" if not self.partial_lex else "N"}\
              \nFree connex: {"Y" if self.free_connex else "N"}\
              \nLEX connex: {"Y" if self.lex_connex else "N"}")

    @staticmethod
    def is_x_connex_cq(hyperedges: List[Set], x_vars: Set|List):
        """
        hyperedges: list of sets of variables, e.g. [{'x','y'}, {'y','z'}]
        x_vars: a set of variables 
        Returns: True if query is X-connex, False otherwise
        """
        if not ConjunctiveQuery.gyo_reduce(hyperedges):
            return False  # original query not acyclic
        extended = hyperedges + [set(x_vars)]
        return ConjunctiveQuery.gyo_reduce(extended)

    @staticmethod
    def gyo_reduce(hyperedges: List[Set]):
        """
        GYO reduction algorithm to test hypergraph acyclicity (robust version).
        Input: hyperedges - list of sets (each set is a hyperedge)
        Returns: True if hypergraph is acyclic, False otherwise
        """
        edges = deepcopy(hyperedges)

        while True:
            changed = False

            # Rule 1: remove all edges that are strict subsets of another
            to_remove = set()
            for i, e1 in enumerate(edges):
                for j, e2 in enumerate(edges):
                    if i != j and e1.issubset(e2):
                        to_remove.add(i)
                        break
            if to_remove:
                edges = [e for idx, e in enumerate(edges) if idx not in to_remove]
                changed = True

            # Rule 2: remove variables that appear only once
            all_vars = [v for edge in edges for v in edge]
            var_count = {v: all_vars.count(v) for v in set(all_vars)}
            new_edges = []
            for edge in edges:
                new_edge = {v for v in edge if var_count[v] > 1}
                if new_edge:
                    new_edges.append(new_edge)
            if new_edges != edges:
                edges = new_edges
                changed = True

            if not changed:
                break

        return len(edges) == 0


    def lex_order_plus_greedy(self):
        """
        Greedy algorithm to safely extend lexicographic order without explicitly checking trios or join trees.
        Algorithm:

        0. Initialize L+:=L.

        1. From the set of atoms, select the atom R that has the largest intersection with variables currently in L+.

        2. Add all free variables in atom R that have not yet been added to L+, and then remove R from the set of atoms.

        3. Repeat steps 1–2 until all free variables have been added to L+.

        Returns:
            lex_ord_plus: List[str], complete lexicographic order safely extended.
        """
        # Remaining atoms (relations)
        remaining_atoms = [set(rel.variables) for rel in self.atoms_free]

        # initial L+
        lex_ord_plus = list(self.lex_order) 
        lex_dict_plus = self.lex_dict
        vars_in_order = set(lex_ord_plus)
        
        # add all free variables
        while len(vars_in_order) < len(self.free_vars):
            # Select atom with maximum intersection with vars_in_order
            max_intersection = -1
            selected_atom = None
            
            for atom in remaining_atoms:
                intersection_size = len(atom & vars_in_order)
                if intersection_size > max_intersection:
                    max_intersection = intersection_size
                    selected_atom = atom
                    
            if selected_atom is None:
                raise ValueError("No suitable atom found; check query connectivity.")
            
            # Add all new variables from selected atom to lex_ord_plus
            new_vars = (selected_atom - vars_in_order)  # sorted for deterministic order
            lex_ord_plus.extend(new_vars)
            vars_in_order.update(new_vars)
            for var in new_vars: 
                lex_dict_plus.update({var: 0})
            
            # Remove selected atom from remaining_atoms
            remaining_atoms.remove(selected_atom)
            
        self.lex_order_plus = lex_ord_plus
        self.lex_dict = lex_dict_plus

    @staticmethod
    def find_ear_and_witness(rels: Set[Relation], root:Relation):
        """
        find an ear and a witness in the set of hyperedges
        return (ear, witness) if an ear exists or return (None, None)
        """
        edges = set(rels)
        root_edge = set(root.variables) if root else None

        # iterate all edges to show mapping from variable to relation
        var_index = defaultdict(list)

        for rel in rels:
            for v in rel.variables:
                var_index[v].append(rel)
        
        # iterate every remaining edge
        for rel in rels: 
            
            # root_edge cannot regarded as an ear
            if root and rel == root: 
                continue

            e = rel.variables
            e_set = set(e)
            # isoloate vertices
            iso_set = set()
            for v in e:
                if len(var_index[v]) == 1:
                    iso_set.add(v)

            join_set = e_set - iso_set
            
            if not join_set:
                # no join vertices
                other_rel = edges - {rel}
                if other_rel:
                    # choose a remaining edge as witness
                    w_rel = next(iter(other_rel)) if not root else root
                    return (rel, w_rel, None)
                else:
                    # no remaining edge
                    return (rel, None, None)
            else: 
                if root_edge and join_set.issubset(root_edge):
                    return (rel, root, join_set)
                # join vertices exist: find witness (containing all join vertices)
                other_rel = edges - {rel}
                for w_rel in other_rel:
                    if join_set.issubset(set(w_rel.variables)):
                        return (rel, w_rel, join_set)
        # No ear found
        return (None, None, None)
    
    @staticmethod
    def ear_decomposition_gyo(atoms:List|Set, root_rel: Relation=None):
        """Determine if the hypergraph is acyclic and construct a join tree if acyclic. 
        """
        relations = set(atoms)

        if root_rel is not None:
            # root_edge = frozenset(root)
            relations.add(root_rel)

        parent_map = {}
        removed_ears = []  
        
        while True:
            if len(relations) <= 1:
                # 1 left could be root edge
                break

            ear, wit, connection = ConjunctiveQuery.find_ear_and_witness(relations, root_rel)
            if ear is None:
                # No ear found: cyclic
                return None
            
            # remove ear if found
            relations.remove(ear)
            removed_ears.append(ear)
            # ear: child. wit: parent
            parent_map[ear] = (wit, connection)
        
        
        # Acyclic and construct a join tree
        all_involved = set(removed_ears)
        # all_involved.update(ed for ed in parent_map.values() if ed is not None)
        all_involved.update(relations)
        
        node_dict = {}
        root_node = None
        # construct every join tree node
        for e in all_involved:
            node_dict[e] = JoinTreeNode(e, aux=True)
        
        # construct the tree structure
        for ear in all_involved: 
            if ear in parent_map.keys():
                p = parent_map[ear][0] 
                con = parent_map[ear][1]
                if p is not None:
                    # ear is connected to the witness
                    node_dict[p].children.append(node_dict[ear])
                    node_dict[p].children_connection[node_dict[ear]] = con
                    node_dict[ear].parent = node_dict[p]
                    node_dict[ear].parent_connection = con

            else: 
                root_node = node_dict[ear]

        return root_node, node_dict.values()
    
    @staticmethod
    def output_subtree(root_node: JoinTreeNode): 
        if root_node is None:
            print("No subtree")
            return 
        
        queue = deque([root_node])

        while queue:
            node = queue.popleft()
            print(f"({node} \n  Children -> {node.children} \n  Parent -> {node.parent}")

            if node.children:      
                for child in node.children:
                    queue.append(child)


    def build_auxiliary_tree(self, is_selection=False): 
        '''
        Auxiliary tree for non-full acycilic cq
        '''
        # join tree with free variables as the root 
        atoms = self.atoms
        free_vars = self.free_vars 
        lex_order = self.lex_dict
        free_rel = Relation('__FREE_REL', free_vars)
        T_prime, T_prime_nodes = self.ear_decomposition_gyo(atoms, root_rel=free_rel) 
        # self.output_subtree(T_prime)

        # join tree projected
        # the relations in the tree are projected with only the free variables 
    
        atoms_proj = [] # relations added due to the projection
        vars_proj_atom = {}  # store the mapping of a tuple of projected variables to a relation
        free_atoms = []

        for atom in atoms: 
            # process each relation
            vars_proj = tuple([v for v in atom.variables if v in free_vars])
            
            if vars_proj: 
                if vars_proj not in vars_proj_atom.keys(): 
                    # new a relation with projection. remember to modify the data
                    if vars_proj != atom.variables:
                        proj_attr_map = [atom.variables.index(attr) for attr in vars_proj] 
                        data_add = Relation.project_remove_duplicates(data=atom.instance_row, 
                                                                        orig_vars=atom.variables, 
                                                                        new_vars=vars_proj)
                        # data_add = [tuple(record[attr] for attr in proj_attr_map) for record in atom.instance_row]
                        # var_proj_out = set(atom.variables).difference(set(vars_proj)) 
                        # data_add = {k: v for k, v in atom.instance_col.items() if k not in var_proj_out}
                        conn = [atom]  
                    else: # project, but use the previous instance
                        data_add = atom.instance_row
                        conn = None
                    atom_add = Relation(f'__{atom.name}_proj', vars_proj, 
                                        connection=conn, instance=data_add, lex_order=lex_order)
                    # print('AAA', atom_add, 'DATA', data_add)
                    free_atoms.append(atom_add)
                        
                    vars_proj_atom[vars_proj] = atom_add
                    atoms_proj.append(atom_add)
                else: 
                    vars_proj_atom[vars_proj].connection.append(atom)

        self.atoms_free = free_atoms 

        T_proj, T_proj_nodes = self.ear_decomposition_gyo(atoms_proj)
        # self.output_subtree(T_proj)
        
        for node_orig in T_prime_nodes: 
            if node_orig.parent and node_orig.parent.parent == None: # neighbors of root
                for node_proj in T_proj_nodes: 
                    # ignore the duplicates
                    if set(node_proj.relation.variables) == set(node_orig.relation.variables): 
                        if node_orig.children: 
                            node_proj.children.extend(node_orig.children)  
                            for child in node_orig.children: 
                                conn = child.parent_connection.intersection(set(node_proj.relation.varibales))
                                child.parent = node_proj
                                child.parent_connection = conn
                                node_proj.children_connection[child] = conn

                    elif is_selection: 
                        continue

                    # connect to the original relation without projection
                    elif node_proj.relation.connection and node_orig.relation in node_proj.relation.connection: 
                        conn = node_proj.relation.variables
                        node_proj.children.append(node_orig)
                        node_proj.children_connection[node_orig] = conn
                        node_orig.parent = node_proj
                        node_orig.parent_connection = conn
        # print("final auxiliary tree")
        
        self.tree_aux = T_proj
        # self.output_subtree(self.tree_aux) 

        # traverse the tree to get new relations
        # self.atoms_free = self.bfs_get_relations(self.tree_aux)
        # self.atoms_free = free_atoms 
        self.semi_join_bottom_up(self.tree_aux)

    @staticmethod
    def semi_join_bottom_up(node: JoinTreeNode):
        if node.children:
        # from leaf nodes
            for child in node.children:
                ConjunctiveQuery.semi_join_bottom_up(child)

            # each child send message to parent: keep which data tuple
            parent_rel = node.relation
            for child, conn in node.children_connection.items():
                if conn:
                    join_attrs  = list(conn)  # e.g. {'y'}
                    parent_rel.semi_join(child.relation, join_attrs)

    def build_join_tree(self): 
        atoms = self.atoms_free
        first_lex = self.lex_order[0]
        for atom in atoms: 
            if first_lex in atom.variables:
                root_relation = atom
                break
        self.root_node, _ = self.ear_decomposition_gyo(atoms, root_rel=root_relation) 
        return self.root_node