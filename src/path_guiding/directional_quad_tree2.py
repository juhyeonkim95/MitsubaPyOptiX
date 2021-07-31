import numpy as np
from path_guiding.quadtree import DTree
import multiprocessing
from pyoptix import Buffer
import copy
from queue import Queue
from utils.timing_utils import *
from multiprocessing import RawArray
from numpy.ctypeslib import ndpointer
import ctypes

# global variable
dtree_index_array = None
dtree_rank_array = None
dtree_depth_array = None
dtree_select_array = None
dtree_value_array = None
current_sizes = None
import invoke
import pathlib

invoke.run(
    "gcc -shared -o libquadtree_updater.so -fPIC quadtree_updater.c"
)
# Load the shared library into ctypes
libname = pathlib.Path().absolute() / "libquadtree_updater.so"
c_lib = ctypes.CDLL(libname)


def get_shared_memory(shape, dtype=np.uint32):
    d = ctypes.c_uint32 if dtype == np.uint32 else ctypes.c_float
    shared_array_base = RawArray(d, int(np.prod(shape)))
    shared_array = np.ctypeslib.as_array(shared_array_base)
    shared_array = shared_array.reshape(shape)
    shared_array.fill(0)
    return shared_array


class DirectionalQuadTree:
    """
    Implement directional quadtrees in SOA (Structure of Arrays) style
    """
    def __init__(self, value_array, max_size=10000, quad_tree_update_type='cpu_single'):
        global dtree_index_array, dtree_rank_array, dtree_depth_array, dtree_select_array, dtree_value_array, current_sizes
        n_s = len(value_array)
        self.n_s = n_s
        self.max_size = max_size
        if quad_tree_update_type == 'cpu_multi':
            dtree_index_array = get_shared_memory((n_s, max_size))
            dtree_rank_array = get_shared_memory((n_s, max_size))
            dtree_depth_array = get_shared_memory((n_s, max_size))
            dtree_select_array = get_shared_memory((n_s, max_size))
            dtree_value_array = get_shared_memory((n_s, max_size), dtype=np.float32)
            np.copyto(dtree_value_array, value_array)
            current_sizes = get_shared_memory((n_s,))
            current_sizes.fill(1)
        else:
            dtree_index_array = np.zeros((n_s, max_size), dtype=np.uint32)
            dtree_rank_array = np.zeros((n_s, max_size), dtype=np.uint32)
            dtree_depth_array = np.zeros((n_s, max_size), dtype=np.uint32)
            dtree_select_array = np.zeros((n_s, max_size), dtype=np.uint32)
            dtree_value_array = value_array
            current_sizes = np.ones((n_s, ), dtype=np.uint32)

    def print(self, index):
        current_size = current_sizes[index]
        print("Size", current_size)
        print("Value array", dtree_value_array[index, 0:current_size])
        print("Index array", dtree_index_array[index, 0:current_size])
        print("Rank array", dtree_rank_array[index, 0:current_size])
        # print("Visit array", self.visit_count_array[0:self.current_size])
        print("Depth array", dtree_depth_array[index, 0:current_size])
        print("Select array", dtree_select_array[index, 0:current_size])

    def get_total_irradiance(self, index):
        irradiance = 0
        area = 0
        current_size = current_sizes[index]
        index_array = dtree_index_array[index]
        select_array = dtree_select_array[index]
        rank_array = dtree_rank_array[index]
        value_array = dtree_value_array[index]
        depth_array = dtree_depth_array[index]

        for i in range(current_size):
            if index_array[i] == 0:
                area_part = pow(0.25, depth_array[i])
                irradiance += value_array[i] * area_part
                area += area_part
        assert area == 1.0
        return irradiance

    def get_total_irradiance_cytpe(self, index):
        fun = c_lib.get_total_irradiance
        fun.restype = ctypes.c_float
        fun.argtypes = [
            ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32
        ]

        return fun(dtree_index_array, dtree_value_array, dtree_depth_array,
                   ctypes.c_uint32(current_sizes[index]), ctypes.c_uint32(self.n_s), ctypes.c_uint32(index))


    def build_rank_array(self, index):
        sum = 0
        current_size = current_sizes[index]
        index_array = dtree_index_array[index]
        select_array = dtree_select_array[index]
        rank_array = dtree_rank_array[index]

        for i in range(current_size):
            rank_array[i] = sum
            sum += index_array[i]

        for i in range(current_size):
            if index_array[i] == 1:
                for j in range(4):
                    child_j = 4 * rank_array[i] + j + 1
                    select_array[child_j] = i

    def update_parent_radiance(self, index):
        current_size = current_sizes[index]
        value_array = dtree_value_array[index]
        index_array = dtree_index_array[index]
        select_array = dtree_select_array[index]

        for i in reversed(range(current_size)):
            if i == 0:
                continue
            parent_id = select_array[i]
            assert index_array[parent_id] == 1
            value_array[parent_id] += value_array[i] * 0.25

    def update_quadtree_single(self, index, threshold=0.01):
        current_size_original = current_sizes[index]
        current_size = current_sizes[index]
        if current_size == self.max_size:
            return

        value_array = dtree_value_array[index]
        depth_array = dtree_depth_array[index]
        index_array = dtree_index_array[index]
        rank_array = dtree_rank_array[index]

        sum = self.get_total_irradiance(index)
        sum2 = self.get_total_irradiance_cytpe(index)

        if sum > 0:
            print("Index", index)
            print("Python irradiance", sum)
            print("CType irradiance", sum2)

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
        current_sizes[index] = new_size

        dtree_index_array[index, 0:new_size] = np.asarray(new_index_array)
        dtree_value_array[index, 0:new_size] = np.asarray(new_value_array)
        dtree_depth_array[index, 0:new_size] = np.asarray(new_depth_array)
        current_sizes[index] = new_size

        self.build_rank_array(index)
        self.update_parent_radiance(index)

    def copy_to_child(self, copy_list):
        for p in copy_list:
            d, s = p
            dtree_index_array[d] = np.copy(dtree_index_array[s])
            dtree_rank_array[d] = np.copy(dtree_rank_array[s])
            dtree_select_array[d] = np.copy(dtree_select_array[s])
            dtree_depth_array[d] = np.copy(dtree_depth_array[s])
            current_sizes[d] = current_sizes[s]

    def refine(self, context, n_s, value_array, quad_tree_update_type='gpu', threshold=0.01):
        if 'cpu' in quad_tree_update_type:
            dtree_value_array[:] = value_array[:]
            # (1) refine quadtree with single thread
            if n_s == 1 or quad_tree_update_type == "cpu_single":
                for i in range(n_s):
                    self.update_quadtree_single(i)

            # (2) refine quadtree with multiprocessing
            elif quad_tree_update_type == "cpu_multi":
                print(n_s, "Multi", multiprocessing.cpu_count())
                n_multiprocess = min(multiprocessing.cpu_count() - 1, n_s)
                with multiprocessing.Pool(n_multiprocess) as p:
                    p.map(self.update_quadtree_single, [_ for _ in range(n_s)])
                # for i in range(n_s):
                #     self.dtrees[i] = results[i]
                #     self.dtree_index_array[i] = self.dtrees[i].index_array
                #     self.dtree_rank_array[i] = self.dtrees[i].rank_array
                #     self.dtree_value_array[i] = self.dtrees[i].value_array
            context['q_table'].copy_from_array(dtree_value_array)
            self.copy_to_context(context)

        # (3) use gpu
        elif quad_tree_update_type == "gpu":
            context.launch(1, n_s)

        zeros = np.zeros((self.n_s, self.max_size), dtype=np.float32)
        zeros2 = np.zeros((self.n_s, self.max_size), dtype=np.uint32)
        context['q_table_accumulated'].copy_from_array(zeros)
        context['q_table_visit_counts'].copy_from_array(zeros2)

    def copy_to_context(self, context):
        context["dtree_index_array"].copy_from_array(dtree_index_array)
        context["dtree_rank_array"].copy_from_array(dtree_rank_array)
        context["dtree_depth_array"].copy_from_array(dtree_depth_array)
        context["dtree_select_array"].copy_from_array(dtree_select_array)
        dtree_current_size_array = np.zeros((self.n_s, ), dtype=np.uint32)
        for i in range(self.n_s):
            dtree_current_size_array[i] = current_sizes[i]
        context["dtree_current_size_array"].copy_from_array(dtree_current_size_array)

    def copy_from_context(self, context):
        context["dtree_index_array"].copy_to_array(dtree_index_array)
        context["dtree_rank_array"].copy_to_array(dtree_rank_array)
        context["dtree_depth_array"].copy_to_array(dtree_depth_array)
        context["dtree_select_array"].copy_to_array(dtree_select_array)
        current_size_array = context["dtree_current_size_array"].to_array()
        for i in range(self.n_s):
            #if current_size_array[i] != self.dtrees[i].current_size:
            #   print("%d Size before %d after %d"% (i, self.dtrees[i].current_size, current_size_array[i]))
            current_sizes[i] = current_size_array[i]

    def create_buffer_to_context(self, context):
        context['dtree_index_array'] = Buffer.from_array(dtree_index_array, buffer_type='io', drop_last_dim=False)
        context['dtree_rank_array'] = Buffer.from_array(dtree_rank_array, buffer_type='io', drop_last_dim=False)
        context['dtree_depth_array'] = Buffer.from_array(dtree_depth_array, buffer_type='io', drop_last_dim=False)
        context['dtree_select_array'] = Buffer.from_array(dtree_select_array, buffer_type='io', drop_last_dim=False)
        current_size_array = np.ones((self.n_s,), dtype=np.uint32)
        context['dtree_current_size_array'] = Buffer.from_array(current_size_array, buffer_type='io', drop_last_dim=False)