from pyoptix import Buffer
from core.utils.math_utils import *
from utils.logging_utils import load_logger

from path_guiding.directional_data_structure.directional_data_structure import *
from path_guiding.directional_data_structure.directional_grid import DirectionalGrid
from path_guiding.directional_data_structure.directional_quad_tree import DirectionalQuadTree

from path_guiding.spatial_data_structure.spatial_data_structure import SpatialDataStructure
from path_guiding.spatial_data_structure.spatial_voxel import SpatialVoxel
from path_guiding.spatial_data_structure.spatial_binary_tree import SpatialAdaptiveBinaryTree
from core.renderer_constants import *


def get_directional_data_structure(q_table, **kwargs) -> DirectionalDataStructure:
    directional_type = kwargs.get("directional_data_structure_type", "grid")
    if directional_type == "grid":
        return DirectionalGrid(q_table, **kwargs)
    elif directional_type == "quadtree" or directional_type == "quad_tree":
        return DirectionalQuadTree(q_table, **kwargs)


def get_spatial_data_structure(q_table, **kwargs) -> SpatialDataStructure:
    spatial_type = kwargs.get("spatial_data_structure_type", "grid")
    if spatial_type == "grid":
        return SpatialVoxel(q_table, **kwargs)
    elif spatial_type == "binary_tree":
        return SpatialAdaptiveBinaryTree(q_table, **kwargs)


class QTable:
    def __init__(self, **kwargs):
        self.radiance_record_logger = load_logger("Radiance Record")
        from utils.timing_utils import time_measure

        self.directional_mapping_method = kwargs.get("directional_mapping_method", "cylindrical")
        self.accumulative_q_table_update = kwargs.get("accumulative_q_table_update", True)

        self.spatial_data_structure = get_spatial_data_structure(self, **kwargs)
        self.directional_data_structure = get_directional_data_structure(self, **kwargs)

        self.radiance_record_logger.info(str(self.spatial_data_structure))
        self.radiance_record_logger.info(str(self.directional_data_structure))

        # Number of states
        self.n_s = self.spatial_data_structure.get_max_size()

        # Number of actions
        self.n_a = self.directional_data_structure.get_size()

        # Make Q Table
        self.q_table = np.zeros((self.n_s, self.n_a), dtype=np.float32)
        self.q_table_accumulated = np.zeros((self.n_s, self.n_a), dtype=np.float32)
        self.q_table_pdf = np.zeros((self.n_s, self.n_a), dtype=np.float32)
        self.q_table_pdf.fill(1 / self.n_a)
        self.q_table_cdf = np.cumsum(self.q_table_pdf, axis=1)
        self.q_table_visit_counts = np.zeros((self.n_s, self.n_a), dtype=np.uint32)
        self.q_table_normal_counts = np.zeros((self.n_s, self.n_a), dtype=np.uint32)
        self.irradiance_table = np.zeros((self.n_s, self.n_a), dtype=np.float32)
        self.max_radiance_table = np.zeros((self.n_s, self.n_a), dtype=np.float32)

        # initial refine
        self.q_table.fill(1)
        self.directional_data_structure.refine(None, self.spatial_data_structure.get_size(), threshold=0.01, force_cpu=True)
        self.q_table.fill(0)

    @staticmethod
    def register_empty_context(context):
        print("Register default value")
        # spatial, grid
        # context['unitCubeNumber'] = np.array([0, 0, 0, 0], dtype=np.uint32)

        # spatial,  octree
        context['stree_index_array'] = Buffer.empty((0,), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
        context['stree_rank_array'] = Buffer.empty((0,), dtype=np.uint32, buffer_type='i', drop_last_dim=False)

        # spatial, adaptive binary tree
        context['stree_visit_count'] = Buffer.empty((0,), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
        context['stree_child_array'] = Buffer.empty((0,), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
        context['stree_parent_array'] = Buffer.empty((0,), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
        context['stree_axis_array'] = Buffer.empty((0,), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
        context['stree_size'] = Buffer.empty((0,), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
        context['stree_leaf_index_array'] = Buffer.empty((0,), dtype=np.uint32, buffer_type='i', drop_last_dim=False)

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
        context['q_table_cdf'] = Buffer.empty((0, 0), dtype=np.float32, buffer_type='io', drop_last_dim=False)
        context['q_table_visit_counts'] = Buffer.empty((0, 0), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
        context['q_table_normal_counts'] = Buffer.empty((0, 0), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
        context['invalid_sample_counts'] = Buffer.empty((0, ), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
        context['valid_sample_counts'] = Buffer.empty((0, ), dtype=np.uint32, buffer_type='i', drop_last_dim=False)

        # etc
        context['irradiance_table'] = Buffer.empty((0, 0), dtype=np.float32, buffer_type='io', drop_last_dim=False)
        context['max_radiance_table'] = Buffer.empty((0, 0), dtype=np.float32, buffer_type='io', drop_last_dim=False)

    def register_to_context(self, context):
        # (1) Spatial
        if isinstance(self.spatial_data_structure, SpatialVoxel):
            context['spatial_data_structure_type'] = np.array(0, dtype=np.uint32)
        elif isinstance(self.spatial_data_structure, SpatialAdaptiveBinaryTree):
            context['spatial_data_structure_type'] = np.array(2, dtype=np.uint32)

        # (2) Directional
        if isinstance(self.directional_data_structure, DirectionalGrid):
            context['directional_data_structure_type'] = np.array(0, dtype=np.uint32)
        elif isinstance(self.directional_data_structure, DirectionalQuadTree):
            context['directional_data_structure_type'] = np.array(1, dtype=np.uint32)

        if self.directional_mapping_method == 'shirley':
            context['directional_mapping_method'] = np.array(0, dtype=np.uint32)
        elif self.directional_mapping_method == 'cylindrical':
            context['directional_mapping_method'] = np.array(1, dtype=np.uint32)

        # spatial, data structure
        self.spatial_data_structure.create_optix_buffer(context)
        self.directional_data_structure.create_optix_buffer(context)

        # q_table
        context['q_table'] = Buffer.from_array(self.q_table, buffer_type='io', drop_last_dim=False)
        context['q_table_accumulated'] = Buffer.from_array(self.q_table_accumulated, buffer_type='io', drop_last_dim=False)
        context['q_table_pdf'] = Buffer.from_array(self.q_table_pdf, buffer_type='io', drop_last_dim=False)
        context['q_table_cdf'] = Buffer.from_array(self.q_table_cdf, buffer_type='io', drop_last_dim=False)
        context['q_table_visit_counts'] = Buffer.from_array(self.q_table_visit_counts, buffer_type='io', drop_last_dim=False)
        context['q_table_normal_counts'] = Buffer.from_array(self.q_table_normal_counts, buffer_type='io', drop_last_dim=False)

        context['irradiance_table'] = Buffer.from_array(self.irradiance_table, buffer_type='io', drop_last_dim=False)
        context['max_radiance_table'] = Buffer.from_array(self.max_radiance_table, buffer_type='io', drop_last_dim=False)

    def copy_to_child(self, copy_list):
        for p in copy_list:
            d, s = p
            self.q_table[d] = self.q_table[s]

    def copy_to_context(self, context):
        context['q_table'].copy_from_array(self.q_table)
        context['q_table_accumulated'].copy_from_array(self.q_table_accumulated)
        context['q_table_pdf'].copy_from_array(self.q_table_pdf)
        context['q_table_cdf'].copy_from_array(self.q_table_cdf)
        context['q_table_visit_counts'].copy_from_array(self.q_table_visit_counts)
        context['q_table_normal_counts'].copy_from_array(self.q_table_normal_counts)

    def copy_from_context(self, context):
        context['q_table'].copy_to_array(self.q_table)
        context['q_table_accumulated'].copy_to_array(self.q_table_accumulated)
        context['q_table_pdf'].copy_to_array(self.q_table_pdf)
        context['q_table_pdf'].copy_to_array(self.q_table_cdf)
        context['q_table_visit_counts'].copy_to_array(self.q_table_visit_counts)
        context['q_table_normal_counts'].copy_to_array(self.q_table_normal_counts)

    def update_pdf(self, context, k=2, **kwargs):
        global updated

        # (0) calculate q table
        if self.accumulative_q_table_update:
            context['q_table_visit_counts'].copy_to_array(self.q_table_visit_counts)
            context['q_table_accumulated'].copy_to_array(self.q_table_accumulated)

            # this fills q_table with right hand value
            self.q_table[:] = np.divide(self.q_table_accumulated, self.q_table_visit_counts,
                                     out=np.zeros_like(self.q_table),
                                     where=self.q_table_visit_counts != 0.0)
            context['q_table'].copy_from_array(self.q_table)
        else:
            context['q_table'].copy_to_array(self.q_table)

        # self.directional_data_structure.visualize(187)

        # (1) Directional Data Structure Refinement
        if isinstance(self.directional_data_structure, DirectionalQuadTree):
            n_s = self.spatial_data_structure.get_size()
            self.directional_data_structure.refine(context, n_s, threshold=0.01)

        # (2) Spatial Data Structure Refinement
        if isinstance(self.spatial_data_structure, SpatialAdaptiveBinaryTree):
            # self.spatial_data_structure.visualize()

            # calc threshold
            c = kwargs.get("binary_tree_split_sample_number", 12000)
            threshold = int(math.pow(2, k / 2) * c)

            # At this moment, radiance info should reside on CPU
            if isinstance(self.directional_data_structure, DirectionalQuadTree) and \
                    self.directional_data_structure.quadtree_update_type=="gpu":
                self.radiance_record_logger.info("Copy From Context!")
                self.directional_data_structure.copy_from_context(context)
                context['q_table'].copy_to_array(self.q_table)

            # make a shallow copy of visited count array
            self.spatial_data_structure.visit_count_array[:] = np.sum(self.q_table_visit_counts, axis=1)

            # Update using C native
            self.spatial_data_structure.refine_native(self.directional_data_structure, self.q_table, threshold)

        # (3) Spatial Data Structure Update to GPU

            # Update data structure to GPU
            self.spatial_data_structure.copy_to_context(context)

        # (4) Directional Data Structure Update to GPU
        if (isinstance(self.directional_data_structure, DirectionalQuadTree) and self.directional_data_structure.quadtree_update_type != 'gpu') or isinstance(self.spatial_data_structure, SpatialAdaptiveBinaryTree):
            context['q_table'].copy_from_array(self.q_table)
            self.directional_data_structure.copy_to_context(context)

        # (5) finally clear q_table accumulated info
        # This is only valid is no data structure update occurs !!!
        if kwargs.get('clear_accumulated_info_per_update', True):
            zeros = np.zeros((self.n_s, self.n_a), dtype=np.float32)
            zeros2 = np.zeros((self.n_s, self.n_a), dtype=np.uint32)
            context['q_table_accumulated'].copy_from_array(zeros)
            context['q_table_visit_counts'].copy_from_array(zeros2)

        epsilon = kwargs.get("epsilon", 0)
        print("EPSILON", epsilon)
        self.q_table += 1e-10
        q_table_sum = np.sum(self.q_table, axis=1, keepdims=True)
        self.q_table_pdf = np.divide(self.q_table, q_table_sum)

        if epsilon > 0:
            # 2. calculate sampling pdf
            # mix with epsilon
            self.q_table_pdf = self.q_table_pdf * (1 - epsilon) + 1 / self.n_a * epsilon

        if isinstance(self.directional_data_structure, DirectionalGrid):
            self.q_table_cdf = np.cumsum(self.q_table_pdf, axis=1)
            context["q_table_cdf"].copy_from_array(self.q_table_cdf)
            context["q_table_pdf"].copy_from_array(self.q_table_pdf)
            context["irradiance_table"].copy_from_array(self.irradiance_table)
            context["max_radiance_table"].copy_from_array(self.max_radiance_table)
