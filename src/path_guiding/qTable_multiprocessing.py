from core.utils.math_utils import *
from pyoptix import Buffer
from path_guiding.quadtree import DTree

import multiprocessing

class QTable:
    def __init__(self, **kwargs):
        self.spatial_type = kwargs.get("spatial_type", "grid")
        self.directional_type = kwargs.get("directional_type", "grid")
        self.directional_mapping_method = kwargs.get("directional_mapping_method", "equal_area")

        self.n_cube = kwargs.get("n_cube", 8)
        self.n_uv = kwargs.get("n_uv", 16)
        self.octree = kwargs.get("octree", None)
        self.max_quatree_count = kwargs.get("max_quadtree_count", 256 * 2)
        print(self.max_quatree_count)

        self.accumulative_q_table_update = kwargs.get("accumulative_q_table_update", True)

        # Number of state
        self.n_s = 0
        if self.spatial_type == "grid":
            self.n_s = self.n_cube * self.n_cube * self.n_cube
        elif self.spatial_type == "octree":
            self.n_s = self.octree.node_number

        # Number of action
        self.n_a = 0
        if self.directional_type == "grid":
            if self.directional_mapping_method == "cylindrical":
                self.n_a = self.n_uv * self.n_uv
            else:
                self.n_a = 2 * self.n_uv * self.n_uv
        elif self.directional_type == "quadtree":
            self.n_a = self.max_quatree_count

        # Make Q Table
        # TODO : change to n_s n_a
        self.q_table = np.zeros((self.n_s, self.n_a), dtype=np.float32)
        self.q_table_accumulated = np.zeros((self.n_s, self.n_a), dtype=np.float32)
        self.q_table_pdf = np.zeros((self.n_s, self.n_a), dtype=np.float32)
        self.q_table_pdf.fill(1 / self.n_a)
        self.q_table_visit_counts = np.zeros((self.n_s, self.n_a), dtype=np.uint32)
        self.q_table_normal_counts = np.zeros((self.n_s, self.n_a), dtype=np.uint32)

        # If directional is quadtree, we need index array
        self.d_trees = []
        self.dtree_index_array = None
        self.dtree_rank_array = None

        if self.directional_type == "quadtree":
            self.dtree_index_array = np.zeros((self.n_s, self.n_a), dtype=np.uint32)
            self.dtree_rank_array = np.zeros((self.n_s, self.n_a), dtype=np.uint32)

            for i in range(self.n_s):
                self.d_trees.append(DTree(i, self.q_table[i], self.q_table_accumulated[i],
                                          self.q_table_visit_counts[i], self.dtree_index_array[i], self.dtree_rank_array[i]))

    def visualize_q_table(self, coord):
        q_id = np.ravel_multi_index(coord, (self.n_cube, self.n_cube, self.n_cube))
        print("Q id", q_id)
        target_radiance_map = self.q_table[q_id]

        if self.directional_type == "quadtree":
            self.d_trees[q_id].visualize_quadtree()
        else:
            if self.directional_mapping_method == "equal_area":
                image = target_radiance_map.reshape(self.n_uv * 2, self.n_uv)
            else:
                image = target_radiance_map.reshape(self.n_uv, self.n_uv)
            return image
            #plt.figure()
            #plt.imshow(image)
            #plt.show()

    def update_quadtree_single(self, i):
        dtree = self.d_trees[i]
        dtree.value_array = self.q_table[i]
        dtree.visit_count_array = self.q_table_visit_counts[i]
        dtree.update(0.01)
        self.q_table[i] = dtree.value_array
        self.q_table_visit_counts[i] = dtree.visit_count_array
        self.dtree_rank_array[i] = dtree.rank_array
        self.dtree_index_array[i] = dtree.index_array

        print(i, "PROCESS")

    def update_quadtree(self, context):
        print("Quad tree update start")
        if self.directional_type == "quadtree":
            self.d_trees[188].visualize_quadtree()
            N_multiprocess = multiprocessing.cpu_count()-1
            with multiprocessing.Pool(N_multiprocess) as p:
                p.map(self.update_quadtree_single, [_ for _ in range(len(self.d_trees))])

            # for i, dtree in enumerate(self.d_trees):
            #     dtree.value_array = self.q_table[i]
            #     dtree.visit_count_array = self.q_table_visit_counts[i]
            #     dtree.update(0.01)

        context['q_table'].copy_from_array(self.q_table)

        zeros = np.zeros((self.n_s, self.n_a), dtype=np.float32)
        zeros2 = np.zeros((self.n_s, self.n_a), dtype=np.uint32)

        context['q_table_accumulated'].copy_from_array(zeros)
        context['q_table_visit_counts'].copy_from_array(zeros2)

        context['dtree_index_array'].copy_from_array(self.dtree_index_array)
        context['dtree_rank_array'].copy_from_array(self.dtree_rank_array)

        print("Quad tree update finished!")

    def update_pdf(self, context, epsilon, cdf=False):
        context['q_table_normal_counts'].copy_to_array(self.q_table_normal_counts)

        if self.accumulative_q_table_update:
            context['q_table_accumulated'].copy_to_array(self.q_table_accumulated)
            context['q_table_visit_counts'].copy_to_array(self.q_table_visit_counts)

            # self.q_table = np.divide(self.q_table_accumulated, self.q_table_visit_counts + 1e-7)
            self.q_table = np.divide(self.q_table_accumulated, self.q_table_visit_counts, out=np.zeros_like(self.q_table),
                                                          where=self.q_table_visit_counts != 0.0)
            #print("Check q_table_accum", np.any(np.isnan(self.q_table_accumulated)))
            #print("Check q_table_visit", np.any(np.isnan(self.q_table_visit_counts)))

            print("Check q", np.any(self.q_table < 0))
            #print(np.min(self.q_table))

            # for i in range(self.n_s):
            #     print("Q sum",i,  np.sum(self.q_table[i]))
            #     print("Visit sum",i, np.sum(self.q_table_visit_counts[i]))

            # self.q_table = np.divide(self.q_table_accumulated, self.q_table_visit_counts, dtype=np.float32)
            context['q_table'].copy_from_array(self.q_table)
        else:
            context['q_table'].copy_to_array(self.q_table)

        if self.directional_type == 'quadtree':
            self.update_quadtree(context)
            return

        print("Q mean", np.mean(self.q_table))
        print("Q max", np.max(self.q_table))

        self.q_table += 1e-6
        q_table_sum = np.sum(self.q_table, axis=1, keepdims=True)
        self.q_table_pdf = np.divide(self.q_table, q_table_sum)

        # std = np.std(self.q_table_pdf, axis=1)
        # t = np.argsort(-std)
        # print("STD", std[t[0:10]])
        # print(std.shape)
        # for i in range(10):
        #     pdf = self.q_table_pdf[:, t[i]]
        #     pdf = -np.sort(-pdf)
        #     print(i, "pdf", pdf[0:15])
        #     print("Sum", np.sum(pdf))
        #     print("Std", std[t[i]])

        self.q_table_pdf = self.q_table_pdf * (1 - epsilon) + 1 / self.n_a * epsilon
        if cdf:
            self.q_table_pdf = np.cumsum(self.q_table_pdf, axis=1)
        context["q_table_pdf"].copy_from_array(self.q_table_pdf)


    @staticmethod
    def register_empty_context(context):
        print("Register default value")
        # spatial, grid
        context['unitCubeNumber'] = np.array([0, 0, 0], dtype=np.uint32)

        # spatial, octree
        context['stree_visit_count'] = Buffer.empty((0,), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
        context['stree_index_array'] = Buffer.empty((0,), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
        context['stree_rank_array'] = Buffer.empty((0,), dtype=np.uint32, buffer_type='i', drop_last_dim=False)

        # directional, grid
        context['unitUVVectors'] = Buffer.empty((0, 3), buffer_type='i', drop_last_dim=True)
        context['unitUVNumber'] = np.array([0, 0], dtype=np.uint32)

        # directional, quadtree
        context['dtree_index_array'] = Buffer.empty((0, 0), buffer_type='i', drop_last_dim=False)
        context['dtree_rank_array'] = Buffer.empty((0, 0), buffer_type='i', drop_last_dim=False)

        # q table
        context['q_table'] = Buffer.empty((0, 0), dtype=np.float32, buffer_type='io', drop_last_dim=False)
        context['q_table_accumulated'] = Buffer.empty((0, 0), dtype=np.float32, buffer_type='io', drop_last_dim=False)
        context['q_table_pdf'] = Buffer.empty((0, 0), dtype=np.float32, buffer_type='io', drop_last_dim=False)
        context['q_table_visit_counts'] = Buffer.empty((0, 0), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
        context['q_table_normal_counts'] = Buffer.empty((0, 0), dtype=np.uint32, buffer_type='i', drop_last_dim=False)

        # etc
        context['irradiance_table'] = Buffer.empty((0, 0), dtype=np.float32, buffer_type='io', drop_last_dim=False)
        context['max_radiance_table'] = Buffer.empty((0, 0), dtype=np.float32, buffer_type='io', drop_last_dim=False)
        mcmc_init = np.random.random((0, 0, 2)).astype(np.float32)
        context['mcmc_table'] = Buffer.from_array(mcmc_init, dtype=np.float32, buffer_type='io', drop_last_dim=True)

    def register_to_context(self, context):
        # config setting
        if self.spatial_type == 'grid':
            context['spatial_table_type'] = np.array(0, dtype=np.uint32)
        elif self.spatial_type == 'octree':
            context['spatial_table_type'] = np.array(1, dtype=np.uint32)

        if self.directional_type == 'grid':
            context['directional_table_type'] = np.array(0, dtype=np.uint32)
        elif self.directional_type == 'quadtree':
            context['directional_table_type'] = np.array(1, dtype=np.uint32)

        if self.directional_mapping_method == 'equal_area':
            context['directional_mapping_method'] = np.array(0, dtype=np.uint32)
        elif self.directional_mapping_method == 'cylindrical':
            context['directional_mapping_method'] = np.array(1, dtype=np.uint32)

        # spatial, grid
        if self.spatial_type == "grid":
            context['unitCubeNumber'] = np.array([self.n_cube] * 3, dtype=np.uint32)

        # spatial, octree
        if self.spatial_type == "octree":
            context['stree_visit_count'] = Buffer.empty((self.octree.total_size, ), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
            context['stree_index_array'] = Buffer.from_array(self.octree.index_array, dtype=np.uint32, buffer_type='i', drop_last_dim=False)
            context['stree_rank_array'] = Buffer.from_array(self.octree.rank_array, dtype=np.uint32, buffer_type='i', drop_last_dim=False)

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
            context['dtree_index_array'] = Buffer.from_array(self.dtree_index_array, buffer_type='i', drop_last_dim=False)
            context['dtree_rank_array'] = Buffer.from_array(self.dtree_rank_array, buffer_type='i', drop_last_dim=False)

        # q_table
        context['q_table'] = Buffer.from_array(self.q_table, buffer_type='io', drop_last_dim=False)
        context['q_table_accumulated'] = Buffer.from_array(self.q_table_accumulated, buffer_type='io', drop_last_dim=False)
        context['q_table_pdf'] = Buffer.from_array(self.q_table_pdf, buffer_type='io', drop_last_dim=False)
        context['q_table_visit_counts'] = Buffer.from_array(self.q_table_visit_counts, buffer_type='io', drop_last_dim=False)
        context['q_table_normal_counts'] = Buffer.from_array(self.q_table_normal_counts, buffer_type='io', drop_last_dim=False)

        # etc
        context['irradiance_table'] = Buffer.empty((self.n_s, self.n_a), dtype=np.float32, buffer_type='io', drop_last_dim=False)
        context['max_radiance_table'] = Buffer.empty((self.n_s, self.n_a), dtype=np.float32, buffer_type='io', drop_last_dim=False)
        mcmc_init = np.random.random((self.n_s, self.n_a, 2)).astype(np.float32)
        context['mcmc_table'] = Buffer.from_array(mcmc_init, dtype=np.float32, buffer_type='io', drop_last_dim=True)

# class TreeQTable(QTable):
#     def __init__(self, index_array, rank_array):
#         super().__init__()
#         self.index_array = index_array
#         self.rank_array = rank_array
#
#
# class TabularQTable(QTable):
#     def __init__(self, type="grid", n_cube=8, n_uv=16):
#         super().__init__()
#         self.unit_cube_number = np.array([n_cube, n_cube, n_cube], dtype=np.uint32)
#         self.state_number = int(np.prod(self.unit_cube_number))
#         self.action_number = n_uv * n_uv * 2
#         self.q_table = np.zeros((self.action_number, self.state_number), dtype=np.float32)
#         self.n_uv = n_uv
#         self.n_cube = n_cube
#
#     def register_to_context(self, context):
#         input_array = np.zeros((self.n_uv * self.n_uv * 2, 3), dtype=np.float32)
#         for i in range(2 * self.n_uv * self.n_uv):
#             v = getDirectionFrom(i, (0.5, 0.5), (self.n_uv, self.n_uv))
#             input_array[i][0] = v[0]
#             input_array[i][1] = v[1]
#             input_array[i][2] = v[2]
#         context['unitUVVectors'] = Buffer.from_array(input_array, dtype=np.float32, buffer_type='i', drop_last_dim=True)
#
#         self.q_table.fill(1e-3)


