import numpy as np
from path_guiding.directional_quad_tree import DirectionalQuadTree
from path_guiding.spatial_binary_tree import SpatialAdaptiveBinaryTree
from pyoptix import Buffer
from core.utils.math_utils import *


class QTable:
    def __init__(self, **kwargs):
        self.spatial_type = kwargs.get("spatial_type", "grid")
        self.directional_type = kwargs.get("directional_type", "grid")
        self.directional_mapping_method = kwargs.get("directional_mapping_method", "equal_area")

        self.n_cube = kwargs.get("n_cube", 8)
        self.n_uv = kwargs.get("n_uv", 16)
        self.octree = kwargs.get("octree", None)

        self.accumulative_q_table_update = kwargs.get("accumulative_q_table_update", True)
        self.spatial_binary_tree = None

        # Number of states
        self.n_s = 0
        if self.spatial_type == "grid":
            self.n_s = self.n_cube * self.n_cube * self.n_cube
        elif self.spatial_type == "octree":
            self.n_s = self.octree.node_number
        elif self.spatial_type == "binary_tree":
            self.n_s = kwargs.get("binary_tree_max_size", 512 * 8)
            self.spatial_binary_tree = SpatialAdaptiveBinaryTree(self.n_s)

        # Number of actions
        self.n_a = 0
        if self.directional_type == "grid":
            if self.directional_mapping_method == "cylindrical":
                self.n_a = self.n_uv * self.n_uv
            else:
                self.n_a = 2 * self.n_uv * self.n_uv
        elif self.directional_type == "quadtree":
            self.n_a = kwargs.get("max_quadtree_count", 256 * 2)

        # Make Q Table
        self.q_table = np.zeros((self.n_s, self.n_a), dtype=np.float32)
        self.q_table_accumulated = np.zeros((self.n_s, self.n_a), dtype=np.float32)
        self.q_table_pdf = np.zeros((self.n_s, self.n_a), dtype=np.float32)
        self.q_table_pdf.fill(1 / self.n_a)
        self.q_table_visit_counts = np.zeros((self.n_s, self.n_a), dtype=np.uint32)
        self.q_table_normal_counts = np.zeros((self.n_s, self.n_a), dtype=np.uint32)
        self.invalid_sample_counts = np.zeros((self.n_s,), dtype=np.uint32)
        self.valid_sample_counts = np.zeros((self.n_s,), dtype=np.uint32)

        if self.directional_type == "quadtree":
            self.directional_quadtree = DirectionalQuadTree(self.q_table, self.n_a, kwargs.get("quad_tree_update_type", 'cpu_single'))

    def visualize_radiance(self, p):
        if self.spatial_type == "binary_tree":
            index = self.spatial_binary_tree.position_to_index(p)
            print("Index", p, index)
            self.directional_quadtree.dtrees[index].visualize_quadtree()
        else:
            self.directional_quadtree.dtrees[188].visualize_quadtree()

    @staticmethod
    def register_empty_context(context):
        print("Register default value")
        # spatial, grid
        context['unitCubeNumber'] = np.array([0, 0, 0], dtype=np.uint32)

        # spatial,  octree
        context['stree_index_array'] = Buffer.empty((0,), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
        context['stree_rank_array'] = Buffer.empty((0,), dtype=np.uint32, buffer_type='i', drop_last_dim=False)

        # spatial, adaptive binary tree
        context['stree_visit_count'] = Buffer.empty((0,), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
        context['stree_child_array'] = Buffer.empty((0,), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
        context['stree_parent_array'] = Buffer.empty((0,), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
        context['stree_axis_array'] = Buffer.empty((0,), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
        context['stree_size'] = Buffer.empty((0,), dtype=np.uint32, buffer_type='i', drop_last_dim=False)

        # directional, grid
        context['unitUVVectors'] = Buffer.empty((0, 3), buffer_type='i', drop_last_dim=True)
        context['unitUVNumber'] = np.array([0, 0], dtype=np.uint32)

        # directional, quadtree
        context['dtree_index_array'] = Buffer.empty((0, 0), buffer_type='i', drop_last_dim=False)
        context['dtree_rank_array'] = Buffer.empty((0, 0), buffer_type='i', drop_last_dim=False)
        context['dtree_depth_array'] = Buffer.empty((0, 0), buffer_type='i', drop_last_dim=False)
        context['dtree_select_array'] = Buffer.empty((0, 0), buffer_type='i', drop_last_dim=False)
        context['dtree_current_size_array'] = Buffer.empty((0,), buffer_type='i', drop_last_dim=False)
        # context['dtree_value_array'] = Buffer.empty((0,), buffer_type='i', drop_last_dim=False)

        # q table
        context['q_table'] = Buffer.empty((0, 0), dtype=np.float32, buffer_type='io', drop_last_dim=False)
        context['q_table_accumulated'] = Buffer.empty((0, 0), dtype=np.float32, buffer_type='io', drop_last_dim=False)
        context['q_table_pdf'] = Buffer.empty((0, 0), dtype=np.float32, buffer_type='io', drop_last_dim=False)
        context['q_table_visit_counts'] = Buffer.empty((0, 0), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
        context['q_table_normal_counts'] = Buffer.empty((0, 0), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
        context['invalid_sample_counts'] = Buffer.empty((0, ), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
        context['valid_sample_counts'] = Buffer.empty((0, ), dtype=np.uint32, buffer_type='i', drop_last_dim=False)

        # etc
        context['irradiance_table'] = Buffer.empty((0, 0), dtype=np.float32, buffer_type='io', drop_last_dim=False)
        context['max_radiance_table'] = Buffer.empty((0, 0), dtype=np.float32, buffer_type='io', drop_last_dim=False)
        mcmc_init = np.random.random((0, 0, 2)).astype(np.float32)
        context['mcmc_table'] = Buffer.from_array(mcmc_init, dtype=np.float32, buffer_type='io', drop_last_dim=True)

    def register_to_context(self, context):
        # (1) Spatial
        if self.spatial_type == 'grid':
            context['spatial_data_structure_type'] = np.array(0, dtype=np.uint32)
        elif self.spatial_type == 'octree':
            context['spatial_data_structure_type'] = np.array(1, dtype=np.uint32)
        elif self.spatial_type == 'binary_tree':
            context['spatial_data_structure_type'] = np.array(2, dtype=np.uint32)

        # (2) Directional
        if self.directional_type == 'grid':
            context['directional_data_structure_type'] = np.array(0, dtype=np.uint32)
        elif self.directional_type == 'quadtree':
            context['directional_data_structure_type'] = np.array(1, dtype=np.uint32)

        if self.directional_mapping_method == 'equal_area':
            context['directional_mapping_method'] = np.array(0, dtype=np.uint32)
        elif self.directional_mapping_method == 'cylindrical':
            context['directional_mapping_method'] = np.array(1, dtype=np.uint32)

        # spatial, grid
        if self.spatial_type == "grid":
            context['unitCubeNumber'] = np.array([self.n_cube] * 3, dtype=np.uint32)

        # spatial, octree
        if self.spatial_type == "octree":
            context['stree_index_array'] = Buffer.from_array(self.octree.index_array, dtype=np.uint32, buffer_type='i', drop_last_dim=False)
            context['stree_rank_array'] = Buffer.from_array(self.octree.rank_array, dtype=np.uint32, buffer_type='i', drop_last_dim=False)

        # spatial, binary tree
        if self.spatial_type == "binary_tree":
            self.spatial_binary_tree.create_buffer_to_context(context)

        # directional, grid
        if self.directional_type == "grid":
            input_array = np.zeros((self.n_uv * self.n_uv * 2, 3), dtype=np.float32)
            for i in range(2 * self.n_uv * self.n_uv):
                v = getDirectionFrom(i, (0.5, 0.5), (self.n_uv, self.n_uv))
                input_array[i][0] = v[0]
                input_array[i][1] = v[1]
                input_array[i][2] = v[2]
            context['unitUVVectors'] = Buffer.from_array(input_array, dtype=np.float32, buffer_type='i', drop_last_dim=True)
            context['unitUVNumber'] = np.array([self.n_uv, self.n_uv], dtype=np.uint32)

        # directional, quadtree
        if self.directional_type == "quadtree":
            self.directional_quadtree.create_buffer_to_context(context)

        # q_table
        context['q_table'] = Buffer.from_array(self.q_table, buffer_type='io', drop_last_dim=False)
        context['q_table_accumulated'] = Buffer.from_array(self.q_table_accumulated, buffer_type='io', drop_last_dim=False)
        context['q_table_pdf'] = Buffer.from_array(self.q_table_pdf, buffer_type='io', drop_last_dim=False)
        context['q_table_visit_counts'] = Buffer.from_array(self.q_table_visit_counts, buffer_type='io', drop_last_dim=False)
        context['q_table_normal_counts'] = Buffer.from_array(self.q_table_normal_counts, buffer_type='io', drop_last_dim=False)
        context['valid_sample_counts'] = Buffer.from_array(self.valid_sample_counts, buffer_type='io', drop_last_dim=False)
        context['invalid_sample_counts'] = Buffer.from_array(self.invalid_sample_counts, buffer_type='io', drop_last_dim=False)


        # etc
        context['irradiance_table'] = Buffer.empty((self.n_s, self.n_a), dtype=np.float32, buffer_type='io', drop_last_dim=False)
        context['max_radiance_table'] = Buffer.empty((self.n_s, self.n_a), dtype=np.float32, buffer_type='io', drop_last_dim=False)
        mcmc_init = np.random.random((self.n_s, self.n_a, 2)).astype(np.float32)
        context['mcmc_table'] = Buffer.from_array(mcmc_init, dtype=np.float32, buffer_type='io', drop_last_dim=True)

    def copy_to_child(self, copy_list):
        for p in copy_list:
            d, s = p
            self.q_table[d] = self.q_table[s]
            #self.q_table_accumulated[d] = self.q_table_accumulated[s]
            #self.q_table_pdf[d] = self.q_table_pdf[s]
            #self.q_table_visit_counts[d] = self.q_table_visit_counts[s]
            #self.q_table_normal_counts[d] = self.q_table_normal_counts[s]

    def copy_to_context(self, context):
        context['q_table'].copy_from_array(self.q_table)
        context['q_table_accumulated'].copy_from_array(self.q_table_accumulated)
        context['q_table_pdf'].copy_from_array(self.q_table_pdf)
        context['q_table_visit_counts'].copy_from_array(self.q_table_visit_counts)
        context['q_table_normal_counts'].copy_from_array(self.q_table_normal_counts)

    def copy_from_context(self, context):
        context['q_table'].copy_to_array(self.q_table)
        context['q_table_accumulated'].copy_to_array(self.q_table_accumulated)
        context['q_table_pdf'].copy_to_array(self.q_table_pdf)
        context['q_table_visit_counts'].copy_to_array(self.q_table_visit_counts)
        context['q_table_normal_counts'].copy_to_array(self.q_table_normal_counts)

    def get_invalid_sample_rate(self, context):
        valid_counts = context['valid_sample_counts'].to_array()
        invalid_counts = context['invalid_sample_counts'].to_array()

        valid_counts_sum = np.sum(valid_counts)
        invalid_counts_sum = np.sum(invalid_counts)

        invalid_rates = invalid_counts_sum / (valid_counts_sum + invalid_counts_sum + 1e-6)
        return invalid_rates

    def update_pdf(self, context, epsilon, cdf=False, quad_tree_update_type='gpu', k=2, **kwargs):
        context['q_table_normal_counts'].copy_to_array(self.q_table_normal_counts)

        if self.spatial_binary_tree is not None:
            self.spatial_binary_tree.copy_from_context(context)
        valid_counts = context['valid_sample_counts'].to_array()
        invalid_counts = context['invalid_sample_counts'].to_array()

        valid_counts_sum = np.sum(valid_counts)
        invalid_counts_sum = np.sum(invalid_counts)

        invalid_rates = valid_counts / (valid_counts + invalid_counts + 1e-6)

        #if self.spatial_binary_tree is not None:
        #    self.spatial_binary_tree.visualize(invalid_rates, self.spatial_binary_tree.visit_count_array)

        invalid_rate = invalid_counts_sum / (valid_counts_sum + invalid_counts_sum + 1e-6)

        print("Total invalid rate : ", invalid_rate)
        # context['valid_sample_counts'].copy_from_array(self.valid_sample_counts)
        # context['invalid_sample_counts'].copy_from_array(self.invalid_sample_counts)

        # 1. calculate q table
        if self.accumulative_q_table_update:
            context['q_table_accumulated'].copy_to_array(self.q_table_accumulated)
            context['q_table_visit_counts'].copy_to_array(self.q_table_visit_counts)
            print("Visit count sum", np.sum(self.q_table_visit_counts))
            print("q_table_accumulated sum", np.sum(self.q_table_accumulated))

            self.q_table[:] = np.divide(self.q_table_accumulated, self.q_table_visit_counts,
                                     out=np.zeros_like(self.q_table),
                                     where=self.q_table_visit_counts != 0.0)
            context['q_table'].copy_from_array(self.q_table)
        else:
            context['q_table'].copy_to_array(self.q_table)

        if self.directional_type == 'quadtree':

            if self.spatial_type == "binary_tree":
                n_s = self.spatial_binary_tree.leaf_node_number
            else:
                n_s = self.n_s
            print("Spatial size", n_s)

            self.directional_quadtree.refine(context, n_s, self.q_table, quad_tree_update_type, threshold=0.01)

            if self.spatial_type == "binary_tree":
                print("Binary tree update start")

                c = kwargs.get("binary_tree_split_sample_number", 12000)
                threshold = int(math.pow(2, k / 2) * c)
                self.spatial_binary_tree.copy_from_context(context)
                self.spatial_binary_tree.visit_count_array[:] = np.sum(self.q_table_visit_counts, axis=1)

                self.spatial_binary_tree.refine_native(self.directional_quadtree, self.directional_quadtree.dtree_value_array, threshold)

                #copy_pairs = self.spatial_binary_tree.refine(threshold, invalid_rates, None)
                #self.directional_quadtree.copy_to_child(copy_pairs)

                self.spatial_binary_tree.copy_to_context(context)

                print("Binary tree update end")

            context['q_table'].copy_from_array(self.directional_quadtree.dtree_value_array)
            self.directional_quadtree.copy_to_context(context)
            zeros = np.zeros((self.n_s, self.n_a), dtype=np.float32)
            zeros2 = np.zeros((self.n_s, self.n_a), dtype=np.uint32)
            context['q_table_accumulated'].copy_from_array(zeros)
            context['q_table_visit_counts'].copy_from_array(zeros2)

            return

        # 2. calculate sampling pdf
        self.q_table += 1e-6
        q_table_sum = np.sum(self.q_table, axis=1, keepdims=True)
        self.q_table_pdf = np.divide(self.q_table, q_table_sum)

        # mix with epsilon
        self.q_table_pdf = self.q_table_pdf * (1 - epsilon) + 1 / self.n_a * epsilon
        if cdf:
            self.q_table_pdf = np.cumsum(self.q_table_pdf, axis=1)
        context["q_table_pdf"].copy_from_array(self.q_table_pdf)

        # if self.spatial_type == "binary_tree":
        #     c = 12000
        #     threshold = math.pow(2, k / 2) * c
        #     self.spatial_binary_tree.refine(threshold)
        #     self.spatial_binary_tree.copy_to_context(context)
