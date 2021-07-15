import numpy as np
from path_guiding.quadtree import DTree
import multiprocessing
from pyoptix import Buffer
import copy


class DirectionalQuadTree:
    """
    Implement directional quadtrees in SOA (Structure of Arrays) style
    """
    def __init__(self, value_array, max_size=10000):
        n_s = len(value_array)
        self.n_s = n_s
        self.dtree_index_array = np.zeros((n_s, max_size), dtype=np.uint32)
        self.dtree_rank_array = np.zeros((n_s, max_size), dtype=np.uint32)
        self.dtree_depth_array = np.zeros((n_s, max_size), dtype=np.uint32)
        self.dtree_select_array = np.zeros((n_s, max_size), dtype=np.uint32)
        self.dtree_value_array = value_array

        self.max_size = max_size

        self.dtrees = []
        for i in range(n_s):
            dtree = DTree(i, self.dtree_value_array[i], self.dtree_index_array[i], self.dtree_rank_array[i])
            self.dtrees.append(dtree)

    def copy_to_child(self, copy_list):
        for p in copy_list:
            d, s = p
            self.dtree_index_array[d] = np.copy(self.dtree_index_array[s])
            self.dtree_rank_array[d] = np.copy(self.dtree_rank_array[s])

            self.dtrees[d].current_size = self.dtrees[s].current_size
            self.dtrees[d].select_array = np.copy(self.dtrees[s].select_array)
            self.dtrees[d].depth_array = np.copy(self.dtrees[s].depth_array)

            # self.dtrees[d] = copy.deepcopy(self.dtrees[s])

    def update_quadtree_single(self, i):
        dtree = self.dtrees[i]
        dtree.value_array = self.dtree_value_array[i]
        dtree.update(0.01)
        return dtree

    def refine(self, context, n_s, value_array, quad_tree_update_type='gpu', threshold=0.01):
        self.dtree_value_array = value_array

        if 'cpu' in quad_tree_update_type:
            # (1) refine quadtree with single thread
            if n_s == 1 or quad_tree_update_type == "cpu_single":
                for i in range(n_s):
                    self.update_quadtree_single(i)

            # (2) refine quadtree with multiprocessing
            elif quad_tree_update_type == "cpu_multi":
                n_multiprocess = min(multiprocessing.cpu_count() - 1, n_s)
                with multiprocessing.Pool(n_multiprocess) as p:
                    results = p.map(self.update_quadtree_single, [_ for _ in range(n_s)])
                for i in range(n_s):
                    self.dtrees[i] = results[i]
                    self.dtree_index_array[i] = self.dtrees[i].index_array
                    self.dtree_rank_array[i] = self.dtrees[i].rank_array
                    self.dtree_value_array[i] = self.dtrees[i].value_array
            context['q_table'].copy_from_array(self.dtree_value_array)
            self.copy_to_context(context)

        # (3) use gpu
        elif quad_tree_update_type == "gpu":
            context.launch(1, n_s)

        zeros = np.zeros((self.n_s, self.max_size), dtype=np.float32)
        zeros2 = np.zeros((self.n_s, self.max_size), dtype=np.uint32)
        context['q_table_accumulated'].copy_from_array(zeros)
        context['q_table_visit_counts'].copy_from_array(zeros2)

    def copy_to_context(self, context):
        context["dtree_index_array"].copy_from_array(self.dtree_index_array)
        context["dtree_rank_array"].copy_from_array(self.dtree_rank_array)
        context["dtree_depth_array"].copy_from_array(self.dtree_depth_array)
        context["dtree_select_array"].copy_from_array(self.dtree_select_array)
        dtree_current_size_array = np.zeros((self.n_s, ), dtype=np.uint32)
        for i in range(self.n_s):
            dtree_current_size_array[i] = self.dtrees[i].current_size
        context["dtree_current_size_array"].copy_from_array(dtree_current_size_array)

    def copy_from_context(self, context):
        context["dtree_index_array"].copy_to_array(self.dtree_index_array)
        context["dtree_rank_array"].copy_to_array(self.dtree_rank_array)
        context["dtree_depth_array"].copy_to_array(self.dtree_depth_array)
        context["dtree_select_array"].copy_to_array(self.dtree_select_array)
        current_size_array = context["dtree_current_size_array"].to_array()
        for i in range(self.n_s):
            #if current_size_array[i] != self.dtrees[i].current_size:
            #   print("%d Size before %d after %d"% (i, self.dtrees[i].current_size, current_size_array[i]))
            self.dtrees[i].current_size = current_size_array[i]

    def create_buffer_to_context(self, context):
        context['dtree_index_array'] = Buffer.from_array(self.dtree_index_array, buffer_type='io', drop_last_dim=False)
        context['dtree_rank_array'] = Buffer.from_array(self.dtree_rank_array, buffer_type='io', drop_last_dim=False)
        context['dtree_depth_array'] = Buffer.from_array(self.dtree_depth_array, buffer_type='io', drop_last_dim=False)
        context['dtree_select_array'] = Buffer.from_array(self.dtree_select_array, buffer_type='io', drop_last_dim=False)
        current_size_array = np.ones((self.n_s,), dtype=np.uint32)
        context['dtree_current_size_array'] = Buffer.from_array(current_size_array, buffer_type='io', drop_last_dim=False)