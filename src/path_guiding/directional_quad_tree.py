import numpy as np
from path_guiding.quadtree import DTree
import multiprocessing
from pyoptix import Buffer
import copy
from queue import Queue
from utils.timing_utils import *
from path_guiding.c_natives import *


class DirectionalQuadTree:
    """
    Implement directional quadtrees in SOA (Structure of Arrays) style
    """
    def __init__(self, value_array, max_size=10000, quad_tree_update_type='cpu_single'):
        n_s = len(value_array)
        self.n_s = n_s
        self.max_size = max_size

        self.dtree_index_array = np.zeros((n_s, max_size), dtype=np.uint32)
        self.dtree_rank_array = np.zeros((n_s, max_size), dtype=np.uint32)
        self.dtree_depth_array = np.zeros((n_s, max_size), dtype=np.uint32)
        self.dtree_select_array = np.zeros((n_s, max_size), dtype=np.uint32)
        self.dtree_value_array = np.copy(value_array).astype(np.float32)
        self.current_sizes = np.ones((n_s, ), dtype=np.uint32)

    def print(self, index):
        current_size = self.current_sizes[index]
        print("Size", current_size)
        print("Value array", self.dtree_value_array[index, 0:current_size])
        print("Index array", self.dtree_index_array[index, 0:current_size])
        print("Rank array", self.dtree_rank_array[index, 0:current_size])
        print("Depth array", self.dtree_depth_array[index, 0:current_size])
        print("Select array", self.dtree_select_array[index, 0:current_size])

    def get_total_irradiance(self, index):
        irradiance = 0
        area = 0
        current_size = self.current_sizes[index]
        index_array = self.dtree_index_array[index]
        value_array = self.dtree_value_array[index]
        depth_array = self.dtree_depth_array[index]

        for i in range(current_size):
            if index_array[i] == 0:
                area_part = pow(0.25, depth_array[i])
                irradiance += value_array[i] * area_part
                area += area_part
        assert area == 1.0
        print("Python Value at 0", value_array[0])
        print("Python Value at 1", value_array[1])
        print("Python Value at 2", value_array[2])
        print("Python Value at 3", value_array[3])

        return irradiance

    def build_rank_array(self, index):
        sum = 0
        current_size = self.current_sizes[index]
        index_array = self.dtree_index_array[index]
        select_array = self.dtree_select_array[index]
        rank_array = self.dtree_rank_array[index]

        for i in range(current_size):
            rank_array[i] = sum
            sum += index_array[i]

        for i in range(current_size):
            if index_array[i] == 1:
                for j in range(4):
                    child_j = 4 * rank_array[i] + j + 1
                    select_array[child_j] = i

    def update_parent_radiance(self, index):
        current_size = self.current_sizes[index]
        value_array = self.dtree_value_array[index]
        index_array = self.dtree_index_array[index]
        select_array = self.dtree_select_array[index]

        for i in reversed(range(current_size)):
            if i == 0:
                continue
            parent_id = select_array[i]
            assert index_array[parent_id] == 1
            value_array[parent_id] += value_array[i] * 0.25

    def update_quadtree_single(self, index, threshold=0.01):
        current_size_original = self.current_sizes[index]
        current_size = self.current_sizes[index]
        if current_size == self.max_size:
            return

        value_array = self.dtree_value_array[index]
        index_array = self.dtree_index_array[index]
        rank_array = self.dtree_rank_array[index]

        sum = self.get_total_irradiance(index)

        if sum > 0:
            print("Index", index)
            print("Python irradiance", sum)

        q = Queue()
        q.put((0, value_array[0], 0))
        new_index_array = []
        new_value_array = []
        new_depth_array = []

        while not q.empty():
            node, val, depth = q.get()
            new_depth_array.append(depth)
            if node < current_size_original:
                # internal node
                if index_array[node] == 1:
                    new_index_array.append(1)
                    new_value_array.append(0)
                    for i in range(4):
                        child_id = 4 * rank_array[node] + i + 1
                        q.put((child_id, value_array[node] / 4, depth + 1))
                # leaf node
                else:
                    if value_array[node] * pow(0.25, depth) >= threshold * sum \
                            and current_size + 4 <= self.max_size:
                        new_index_array.append(1)
                        for i in range(4):
                            q.put((current_size + i, value_array[node] / 4, depth + 1))
                        new_value_array.append(0)
                        current_size += 4
                    else:
                        new_index_array.append(0)
                        new_value_array.append(value_array[node])
            else:
                new_index_array.append(0)
                new_value_array.append(val)

        new_size = len(new_index_array)

        self.dtree_index_array[index, 0:new_size] = np.asarray(new_index_array)
        self.dtree_value_array[index, 0:new_size] = np.asarray(new_value_array)
        self.dtree_depth_array[index, 0:new_size] = np.asarray(new_depth_array)
        self.current_sizes[index] = new_size

        self.build_rank_array(index)
        self.update_parent_radiance(index)

    def copy_to_child(self, copy_list):
        for p in copy_list:
            d, s = p
            self.dtree_index_array[d] = np.copy(self.dtree_index_array[s])
            self.dtree_rank_array[d] = np.copy(self.dtree_rank_array[s])
            self.dtree_select_array[d] = np.copy(self.dtree_select_array[s])
            self.dtree_depth_array[d] = np.copy(self.dtree_depth_array[s])
            self.dtree_value_array[d] = np.copy(self.dtree_value_array[s])
            self.current_sizes[d] = self.current_sizes[s]

    def refine(self, context, n_s, value_array, quad_tree_update_type='gpu', threshold=0.01):
        if 'cpu' in quad_tree_update_type:
            self.dtree_value_array[:] = value_array[:]
            # (1) refine quadtree with single thread
            if n_s == 1 or quad_tree_update_type == "cpu_single":
                update_quadtree_native(pp(self.dtree_index_array), pp(self.dtree_rank_array), pp(self.dtree_depth_array),
                                       pp(self.dtree_select_array), pp(self.dtree_value_array),
                                       self.current_sizes, self.n_s)

            # (2) refine quadtree with multiprocessing
            elif quad_tree_update_type == "cpu_multi":
                update_quadtree_multi_native(pp(self.dtree_index_array), pp(self.dtree_rank_array), pp(self.dtree_depth_array),
                                             pp(self.dtree_select_array), pp(self.dtree_value_array),
                                             self.current_sizes, self.n_s)

            #context['q_table'].copy_from_array(self.dtree_value_array)
            #self.copy_to_context(context)

        # (3) use gpu
        elif quad_tree_update_type == "gpu":
            context.launch(1, n_s)


    def copy_to_context(self, context):
        context["dtree_index_array"].copy_from_array(self.dtree_index_array)
        context["dtree_rank_array"].copy_from_array(self.dtree_rank_array)
        context["dtree_depth_array"].copy_from_array(self.dtree_depth_array)
        context["dtree_select_array"].copy_from_array(self.dtree_select_array)
        dtree_current_size_array = np.zeros((self.n_s, ), dtype=np.uint32)
        for i in range(self.n_s):
            dtree_current_size_array[i] = self.current_sizes[i]
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
            self.current_sizes[i] = current_size_array[i]

    def create_buffer_to_context(self, context):
        context['dtree_index_array'] = Buffer.from_array(self.dtree_index_array, buffer_type='io', drop_last_dim=False)
        context['dtree_rank_array'] = Buffer.from_array(self.dtree_rank_array, buffer_type='io', drop_last_dim=False)
        context['dtree_depth_array'] = Buffer.from_array(self.dtree_depth_array, buffer_type='io', drop_last_dim=False)
        context['dtree_select_array'] = Buffer.from_array(self.dtree_select_array, buffer_type='io', drop_last_dim=False)
        current_size_array = np.ones((self.n_s,), dtype=np.uint32)
        context['dtree_current_size_array'] = Buffer.from_array(current_size_array, buffer_type='io', drop_last_dim=False)