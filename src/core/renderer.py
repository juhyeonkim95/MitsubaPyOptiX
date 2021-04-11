from pyoptix import Context, Compiler, Program, Material, \
    EntryPoint
from pyoptix.enums import ExceptionType
from core.scene import Scene
from core.renderer_utils import *
from core.camera import Camera
import time
from core.utils.math_utils import *
from os.path import dirname
import matplotlib.pyplot as plt
from datetime import timedelta
import open3d as o3d
from core.qTable import QTable

from core.quadtree import *

SAMPLE_UNIFORM = 0
SAMPLE_COSINE = 1
SAMPLE_HG = 1
SAMPLE_Q_PROPORTION = 2
SAMPLE_Q_COS_PROPORTION = 3
SAMPLE_Q_COS_MCMC= 4
SAMPLE_Q_COS_REJECT= 5
SAMPLE_Q_SPHERE=6
SAMPLE_Q_QUADTREE=7

SAMPLE_Q_HG_PROPORTION = 3


Q_UPDATE_EXPECTED_SARSA = 0
Q_UPDATE_Q_LEARNING = 1
Q_UPDATE_SARSA = 2
Q_UPDATE_MONTE_CARLO = 3

Q_SAMPLE_PROPORTIONAL_TO_Q = 1
Q_SAMPLE_PROPORTIONAL_TO_Q_SQUARE = 2


class Renderer:
    def __init__(self, scale=1, force_all_diffuse=False):
        self.scene = None
        self.scene_name = None
        self.context = None
        self.entry_point = None
        self.quad_tree_updater_entry_point = None

        self.program_dictionary = {}
        self.material_dict = {}
        self.width = 0
        self.height = 0

        self.scene_epsilon = 1e-3
        self.scale = scale
        self.force_all_diffuse=force_all_diffuse
        self.reference_image = None
        self.scene_octree = None

    def create_context(self):
        context = Context()
        context.set_ray_type_count(2)
        context.set_entry_point_count(1)
        context.set_stack_size(2000)

        context['scene_epsilon'] = np.array(self.scene_epsilon, dtype=np.float32)
        context['pathtrace_ray_type'] = np.array(0, dtype=np.uint32)
        context['pathtrace_shadow_ray_type'] = np.array(1, dtype=np.uint32)
        context['bad_color'] = np.array([1000000., 0., 1000000.], dtype=np.float32)
        context['bg_color'] = np.zeros(3, dtype=np.float32)
        self.context = context

        #context.set_print_enabled(True)
        #context.set_exception_enabled(ExceptionType.all, True)
        #context.set_usage_report_callback(3)

        # print("get_cpu_num_of_threads", context.get_cpu_num_of_threads())
        # print("get_stack_size", context.get_stack_size())
        # print("get_available_devices_count", context.get_available_devices_count())
        # print("get_enabled_device_count", context.get_enabled_device_count())
        # print("get_used_host_memory", context.get_used_host_memory())
        # print("get_available_device_memory", context.get_available_device_memory(0))

    def init_entry_point(self, has_envmap):
        if has_envmap:
            miss_program = self.program_dictionary["miss_envmap"]
        else:
            miss_program = self.program_dictionary["miss"]

        self.context.set_ray_generation_program(0, self.program_dictionary['ray_generation'])
        self.context.set_exception_program(0, self.program_dictionary['exception'])
        self.context.set_miss_program(0, miss_program)

        # self.context.set_ray_generation_program(1, self.program_dictionary['quad_tree_updater'])
        # self.entry_point = EntryPoint(self.program_dictionary['ray_generation'],
        #                          self.program_dictionary['exception'],
        #                          miss_program)
        # self.quad_tree_updater_entry_point = EntryPoint()

    def init_programs(self):
        #Compiler.clean()
        #Compiler.keep_device_function = False

        program_dictionary = {}
        # renderer
        target_program = 'optix/integrators/path_trace_camera.cu'
        program_dictionary["ray_generation"] = Program(target_program, 'pathtrace_camera')
        program_dictionary["exception"] = Program(target_program, 'exception')

        program_dictionary["miss_envmap"] = Program('optix/integrators/miss_program.cu', 'miss_environment_mapping')
        program_dictionary["miss"] = Program('optix/integrators/miss_program.cu', 'miss')

        # Geometries
        program_dictionary['tri_mesh_bb'] = Program('optix/shapes/triangle_mesh.cu', 'mesh_bounds')
        program_dictionary['tri_mesh_it'] = Program('optix/shapes/triangle_mesh.cu', 'mesh_intersect_refine')

        program_dictionary["quad_bb"] = Program('optix/shapes/parallelogram.cu', 'bounds')
        program_dictionary["quad_it"] = Program('optix/shapes/parallelogram.cu', 'intersect')
        program_dictionary["sphere_bb"] = Program('optix/shapes/sphere.cu', 'bounds')
        program_dictionary["sphere_it"] = Program('optix/shapes/sphere.cu', 'robust_intersect')
        program_dictionary["disk_bb"] = Program('optix/shapes/disk.cu', 'bounds')
        program_dictionary["disk_it"] = Program('optix/shapes/disk.cu', 'intersect')

        # Materials
        closest_hit_program = 'optix/integrators/hit_program.cu'
        any_hit_hit_program = 'optix/integrators/any_hit_program.cu'

        program_dictionary["closest_hit"] = Program(closest_hit_program, 'closest_hit')
        program_dictionary["closest_hit_light"] = Program('optix/integrators/light_hit_program.cu', 'diffuseEmitter')
        program_dictionary["any_hit_cutout"] = Program(any_hit_hit_program, 'any_hit_cutout')

        program_dictionary["any_hit_shadow"] = Program(any_hit_hit_program, 'any_hit_shadow')
        program_dictionary["any_hit_shadow_cutout"] = Program(any_hit_hit_program, 'any_hit_shadow_cutout')

        # program_dictionary['quad_tree_updater'] = Program('optix/q_table/quad_tree_updater.cu', 'quad_tree_updater')

        self.program_dictionary = program_dictionary

    def init_materials(self):
        material_dict = {}
        program_dictionary = self.program_dictionary
        print(program_dictionary)
        opaque_material = Material()
        opaque_material.set_closest_hit_program(0, program_dictionary["closest_hit"])
        opaque_material.set_any_hit_program(1, program_dictionary["any_hit_shadow"])

        cutout_material = Material()
        cutout_material.set_closest_hit_program(0, program_dictionary["closest_hit"])
        cutout_material.set_any_hit_program(0, program_dictionary["any_hit_cutout"])
        cutout_material.set_any_hit_program(1, program_dictionary["any_hit_shadow_cutout"])

        light_material = Material()
        light_material.set_closest_hit_program(0, program_dictionary["closest_hit_light"])
        light_material.set_any_hit_program(1, program_dictionary["any_hit_shadow"])

        material_dict['opaque_material'] = opaque_material
        material_dict['cutout_material'] = cutout_material
        material_dict['light_material'] = light_material
        self.material_dict = material_dict

    def init_scene(self):
        scene = self.scene
        context = self.context

        scene.optix_load_images()
        scene.optix_create_objs(self.program_dictionary)
        scene.optix_create_geometry_instances(self.program_dictionary, self.material_dict, self.force_all_diffuse)
        create_geometry(context, scene)
        create_scene_lights(context, scene)
        create_scene_materials(context, scene)

    def init_spherical_camera(self, pos, size, map_type):
        room_size = self.scene.bbox.bbox_max - self.scene.bbox.bbox_min
        self.context["spherical_cam_position"] = pos * room_size + self.scene.bbox.bbox_min
        self.context["spherical_cam_size"] = size * room_size
        self.context["spherical_cam_directional_mapping"] = np.array(map_type, dtype=np.uint32)
        self.context["camera_type"] = np.array(1, dtype=np.uint32)

    def init_camera(self):
        camera = self.scene.camera
        context = self.context
        fov = camera.fov
        aspect_ratio = float(self.width) / float(self.height)
        fovx = (camera.fov_axis == 'x')

        # calculate camera variables
        W = np.array(camera.w)
        U = np.array(camera.u)
        V = np.array(camera.v)
        wlen = np.sqrt(np.sum(W ** 2))
        if fovx:
            ulen = wlen * math.tan(0.5 * fov * math.pi / 180)
            U *= ulen
            vlen = ulen / aspect_ratio
            V *= vlen
        else:
            vlen = wlen * math.tan(0.5 * fov * math.pi / 180)
            V *= vlen
            ulen = vlen * aspect_ratio
            U *= ulen

        context["eye"] = camera.eye
        context["U"] = U
        context["V"] = V
        context["W"] = W
        context["focalDistance"] = np.array(5, dtype=np.float32)
        # context["apertureRadius"] = np.array(0.5, dtype=np.float32)
        context["camera_type"] = np.array(0, dtype=np.uint32)

    def init_scene_config(self, scene_name):
        # load scene info (non optix)
        self.scene = Scene(scene_name)
        self.scene.load_scene_from("../scene/%s/scene.xml" % scene_name)
        self.width = self.scene.width // self.scale
        self.height = self.scene.height // self.scale

    def init_optix(self, scene_name):
        self.create_context()
        self.init_programs()
        self.init_entry_point(self.scene.has_envmap)

        self.init_materials()
        self.init_scene()
        self.init_camera()

    def prepare_scene(self, scene_name):
        pass

    def register_context_related(self):
        context = self.context
        height = self.height
        width = self.width
        context['hit_count_buffer'] = Buffer.empty((height, width, 1), dtype=np.float32, buffer_type='o', drop_last_dim=True)
        context['path_length_buffer'] = Buffer.empty((height, width, 1), dtype=np.float32, buffer_type='o', drop_last_dim=True)
        context['scatter_type_buffer'] = Buffer.empty((height, width, 2), dtype=np.float32, buffer_type='o', drop_last_dim=True)
        context['output_buffer'] = Buffer.empty((height, width, 4), dtype=np.float32, buffer_type='o', drop_last_dim=True)
        context['output_buffer2'] = Buffer.empty((height, width, 4), dtype=np.float32, buffer_type='o',  drop_last_dim=True)

    def construct_stree(
            self,
            scene_name="cornell-box",
            max_path_depth=2,
            max_octree_depth=3
    ):
        self.scene_epsilon = 1e-3 if scene_name == "veach_door_simple" else 1e-5

        if self.scene_name != scene_name:
            self.scene_name = scene_name
            self.init_scene_config(scene_name)
            self.init_optix(scene_name)
        else:
            print("Skipped because already prepared")

        context = self.context
        scene = self.scene
        width = self.width
        height = self.height

        context['rr_begin_depth'] = np.array(max_path_depth, dtype=np.uint32)
        context['max_depth'] = np.array(max_path_depth, dtype=np.uint32)
        context["sample_type"] = np.array(SAMPLE_COSINE, dtype=np.uint32)
        context["use_mis"] = np.array(0, dtype=np.uint32)
        context["samples_per_pass"] = np.array(1, dtype=np.uint32)
        context['construct_stree'] = np.array(1, dtype=np.uint32)
        context['point_buffer'] = Buffer.empty((max_path_depth, height, width, 3), dtype=np.float32, buffer_type='o', drop_last_dim=True)
        QTable.register_empty_context(context)
        self.register_context_related()

        # create_q_table_related(context, scene.bbox.bbox_max, height, width, self.n_cube, self.uv_n, self.scene_octree)
        context.validate()
        context.compile()
        context.launch(0, width, height, 1)

        #self.entry_point.launch((self.width, self.height, 1), context)
        pos_buffer = context['point_buffer'].to_array()
        print(pos_buffer.shape)
        xyz = pos_buffer.reshape((width * height * max_path_depth, 3))
        xyz = (xyz - scene.bbox.bbox_min) / (scene.bbox.bbox_max - scene.bbox.bbox_min)

        #tree = ot.PyOctree(xyz)

        start_time = time.time()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        octree = o3d.geometry.Octree(max_depth=max_octree_depth)
        octree.convert_from_point_cloud(pcd)
        #octree.traverse(f_traverse)

        print("Octree build time", time.time() - start_time)
        print(octree.locate_leaf_node(pcd.points[0]))
        #o3d.visualization.draw_geometries([pcd], mesh_show_wireframe=False)
        #o3d.visualization.draw_geometries([octree], mesh_show_wireframe=True)
        index_array = octree_to_index_array(octree)
        #index_array = [1] * (1 + 8 + 64) + [0] * (8 * 64)
        #index_array = np.array(index_array, dtype=np.uint32)
        self.scene_octree = Octree(index_array)

    def render(
            self,
            scene_name="cornell-box",
            spp=256,
            time_limit_in_sec=-1,
            time_limit_init_ignore_step=0,
            sample_type=SAMPLE_COSINE,
            q_table_update_method=Q_UPDATE_MONTE_CARLO,
            q_value_sample_method=1,
            q_value_sample_constant=1.0,
            show_q_value_map=False,
            show_picture=False,
            save_image=False,
            save_image_name=None,
            scatter_sample_type=1,
            use_mis=False,
            use_tone_mapping=True,
            use_soft_q_update=False,
            samples_per_pass=16,
            learning_method="incremental",
            sample_combination="none",
            accumulative_q_table_update=True,
            export_video=False,
            export_video_name=None,
            save_q_cos=False,
            do_iteratively=False,
            max_depth=8,
            rr_begin_depth=4,
            uv_n=8,
            n_cube=32,
            spatial_type='grid',
            directional_type='grid',
            directional_mapping_method="equal_area",
            scene_epsilon=1e-3,
            use_brdf_first_force=True,
            force_update_q_table=False,
            spherical_pos=None,
            spherical_size=None,
            spherical_map_type=0
    ):
        self.scene_epsilon = 1e-3 if scene_name == "veach_door_simple" else 1e-5
        is_budget_time = (time_limit_in_sec > 0)
        start_time = time.time()

        if self.scene_name != scene_name:
            self.scene_name = scene_name
            self.init_scene_config(scene_name)
            self.init_optix(scene_name)
            print("Optix program Prepare Time:", str(timedelta(seconds=time.time() - start_time)))
        else:
            print("Skipped because already prepared")
        start_time = time.time()

        self.width = self.scene.width // self.scale
        self.height = self.scene.height // self.scale

        if spherical_pos is not None:
            if spherical_map_type == 0:
                self.width *= 2
            self.init_spherical_camera(spherical_pos, spherical_size, spherical_map_type)
        else:
            self.init_camera()

        context = self.context
        scene = self.scene
        width = self.width
        height = self.height



        context["sigma_s"] = np.array(0, dtype=np.float32)
        context["sigma_a"] = np.array(0, dtype=np.float32)
        context["sigma_t"] = np.array(0, dtype=np.float32)
        context["hg_g"] = np.array(0, dtype=np.float32)

        context['rr_begin_depth'] = np.array(rr_begin_depth, dtype=np.uint32)
        context['max_depth'] = np.array(max_depth, dtype=np.uint32)
        print("max_depth", max_depth)
        context["sample_type"] = np.array(sample_type, dtype=np.uint32)
        context["use_mis"] = np.array(1 if use_mis else 0, dtype=np.uint32)
        context["use_tone_mapping"] = np.array(1 if use_tone_mapping else 0, dtype=np.uint32)
        context["use_soft_q_update"] = np.array(1 if use_soft_q_update else 0, dtype=np.uint32)
        context["save_q_cos"] = np.array(1 if save_q_cos else 0, dtype=np.uint32)

        context["scatter_sample_type"] = np.array(scatter_sample_type, dtype=np.uint32)
        context["q_table_update_method"] = np.array(q_table_update_method, dtype=np.uint32)
        context["q_value_sample_method"] = np.array(q_value_sample_method, dtype=np.uint32)
        context["q_value_sample_constant"] = np.array(q_value_sample_constant, dtype=np.float32)
        context["samples_per_pass"] = np.array(1, dtype=np.uint32)
        context["accumulative_q_table_update"] = np.array(1 if accumulative_q_table_update else 0, dtype=np.uint32)
        context['sqrt_num_samples'] = np.array(int(math.sqrt(spp)), dtype=np.uint32)
        context['construct_stree'] = np.array(0, dtype=np.uint32)
        context['point_buffer'] = Buffer.empty((0, 0, 0, 3), dtype=np.float32, buffer_type='o',
                                               drop_last_dim=True)

        dont_need_q_table_update = (sample_type == SAMPLE_UNIFORM) or (sample_type == SAMPLE_COSINE)
        need_q_table_update = not dont_need_q_table_update
        need_q_table_update = need_q_table_update or force_update_q_table

        context["need_q_table_update"] = np.array(1 if need_q_table_update else 0, dtype=np.uint32)

        room_size = scene.bbox.bbox_max - scene.bbox.bbox_min
        context['scene_bbox_min'] = scene.bbox.bbox_min
        context['scene_bbox_max'] = scene.bbox.bbox_max
        context['scene_bbox_extent'] = room_size

        self.register_context_related()
        QTable.register_empty_context(context)

        q_table = None
        if need_q_table_update:
            q_table = QTable(n_cube=n_cube, n_uv=uv_n, spatial_type=spatial_type,  directional_type=directional_type,
                             directional_mapping_method=directional_mapping_method,
                             accumulative_q_table_update=accumulative_q_table_update,
                             octree=self.scene_octree)
            q_table.register_to_context(context)

        #n_a = self.uv_n * self.uv_n * 2
        #q_table = np.zeros((n_a, n_s), dtype=np.float32)
        #equal_table = np.zeros((n_a, n_s), dtype=np.float32)
        #equal_table.fill(1 / n_a)

        # initial samples
        if learning_method == "exponential":
            current_samples_per_pass = 1
        else:
            current_samples_per_pass = samples_per_pass
            if samples_per_pass == -1:
                current_samples_per_pass = spp

        hit_sum = 0
        hit_counts = []
        elapsed_times = []
        elapsed_times_single_sample = []

        left_samples = spp
        completed_samples = 0
        n_pass = 0
        inv_variance_weights = []
        output_images = []
        need_to_save_intermediate_outputs = False

        final_image = np.zeros((height, width, 4))
        q_table_update_elapsed_time_accumulated = 0

        is_first = True

        print("Optix Prepare Time:", str(timedelta(seconds=time.time() - start_time)))

        start_time = time.time()
        start_time_2 = None
        while True:
            if is_budget_time:
                if n_pass > time_limit_init_ignore_step:
                    if time.time() - start_time_2 > time_limit_in_sec:
                        break
                elif n_pass == time_limit_init_ignore_step:
                    start_time_2 = time.time()
            else:
                if left_samples <= 0:
                    break

            if do_iteratively:
                a = current_samples_per_pass
                b = 1
            else:
                a = 1
                b = current_samples_per_pass

            context["samples_per_pass"] = np.array(a, dtype=np.uint32)
            print("Current Pass: %d, Current Samples: %d" % (n_pass, current_samples_per_pass))

            if sample_type != 4444:#SAMPLE_Q_SPHERE:
                epsilon = getEpsilon(completed_samples, 100000, t=1, k=100)
            else:
                epsilon = 0.0

            context["frame_number"] = np.array((completed_samples + 1), dtype=np.uint32)

            # Run OptiX program
            ith_start_time = time.time()
            if is_first:
                if use_brdf_first_force:
                    context['is_first_pass'] = np.array(1, dtype=np.uint32)
                else:
                    context['is_first_pass'] = np.array(0, dtype=np.uint32)
                # self.entry_point.launch((width, height, b), context)
                context.validate()
                context.compile()
                context.launch(0, width, height, b)
                is_first = False
            else:
                context['is_first_pass'] = np.array(0, dtype=np.uint32)
                context.launch(0, width, height, b)

            ith_end_time = time.time()
            ith_time = (ith_end_time - ith_start_time)
            ith_time_single_sample = ith_time / current_samples_per_pass
            print("Ith time", ith_time_single_sample)
            elapsed_times.append(ith_time)
            elapsed_times_single_sample.append(ith_time_single_sample)

            #start_time_3 = time.time()
            #total_path_length_sum = np.sum(context['path_length_buffer'].to_array())
            #visit_counts = context['visit_counts'].to_array()
            #total_visited_counts_sum = np.sum(visit_counts)
            #temp_difference = total_path_length_sum - total_visited_counts_sum
            #print("path minus visited counts", temp_difference)
            #print("path cal time", time.time() - start_time_3)

            if need_q_table_update:
                q_table_update_start_time = time.time()
                q_table.update_pdf(context, epsilon, sample_type is SAMPLE_Q_SPHERE)

                # if accumulative_q_table_update:
                #     visit_counts = context['visit_counts'].to_array()
                #     q_table_accumulated = context['q_table_accumulated'].to_array()
                #     q_table = np.divide(q_table_accumulated, visit_counts, out=np.zeros_like(q_table),
                #                         where=visit_counts != 0.0)
                #     print("Check q", np.any(q_table<0))
                #     context['q_table'].copy_from_array(q_table)
                # else:
                #     context["q_table"].copy_to_array(q_table)
                #
                context['irradiance_table'] = Buffer.empty((q_table.n_s, q_table.n_a), dtype=np.float32, buffer_type='io',
                                                           drop_last_dim=False)
                context['max_radiance_table'] = Buffer.empty((q_table.n_s, q_table.n_a), dtype=np.float32, buffer_type='io',
                                                             drop_last_dim=False)
                #
                # q_table += 1e-6
                # q_table_sum = np.sum(q_table, axis=0, keepdims=True)
                #
                # policy_table = np.divide(q_table, q_table_sum)
                # policy_table = policy_table * (1 - epsilon) + equal_table * epsilon
                # context['q_table_old'].copy_from_array(policy_table)

                q_table_update_elapsed_time = time.time() - q_table_update_start_time
                q_table_update_elapsed_time_accumulated += q_table_update_elapsed_time
            np_hit_count = context['hit_count_buffer'].to_array()
            hit_new_sum = np.sum(np_hit_count)

            if n_pass > 0:
                hit = hit_new_sum - hit_sum
                hit_counts.append(hit / current_samples_per_pass)

            hit_sum = hit_new_sum
            print("Hit sum", hit_sum)

            if need_to_save_intermediate_outputs:
                output_image = context['output_buffer'].to_array()
                output_image /= current_samples_per_pass
                # if use inverse variance stack images
                if sample_combination == "inverse_variance":
                    output2_image = context['output_buffer2'].to_array()
                    output2_image /= current_samples_per_pass

                    variance = output2_image - output_image * output_image
                    variance_lum = np.clip(get_luminance(variance), 1e-4, 10000)
                    inv_variance = 1 / np.mean(variance_lum)
                    # inv_variance = np.mean(1 / variance_lum)
                    inv_variance_weights.append(inv_variance)
                elif sample_combination == "discard":
                    final_image = output_image
                else:
                    final_image += current_samples_per_pass * output_image

                if sample_combination == "inverse_variance" or export_video:
                    output_images.append(output_image)

                context['output_buffer'] = Buffer.empty((height, width, 4), dtype=np.float32, buffer_type='o',
                                                        drop_last_dim=True)
                context['output_buffer2'] = Buffer.empty((height, width, 4), dtype=np.float32, buffer_type='o',
                                                         drop_last_dim=True)

            completed_samples += current_samples_per_pass
            if not is_budget_time:
                # update next pass
                left_samples -= current_samples_per_pass
                if learning_method == "exponential":
                    current_samples_per_pass *= 2
                    if left_samples - current_samples_per_pass < 2 * current_samples_per_pass:
                        current_samples_per_pass = left_samples
                else:
                    current_samples_per_pass = samples_per_pass
                current_samples_per_pass = min(current_samples_per_pass, left_samples)
            n_pass += 1
        end_time = time.time()

        if sample_combination == "inverse_variance":
            final_image = np.zeros_like(output_images[0])
            accumulated_weight = 0
            for _ in range(4):
                weight = inv_variance_weights.pop()
                print("weight", weight)
                final_image += weight * output_images.pop()
                accumulated_weight += weight
            final_image /= accumulated_weight
        else:
            output_image = context['output_buffer'].to_array()
            final_image = output_image / completed_samples

        print(final_image.shape)
        final_image = np.flipud(final_image)

        hdr_image = final_image[:, :, 0:3]

        print(np.mean(hdr_image), "Image MEAN")

        # hdr_image = cv.cvtColor(hdr_image, cv.COLOR_RGB2BGR)
        # tone_map = cv.createTonemapReinhard(gamma=2.2, intensity=0, light_adapt=0, color_adapt=0)
        # ldr_image = tone_map.process(hdr_image)
        # ldr_image = cv.cvtColor(ldr_image, cv.COLOR_RGB2BGR)
        ldr_image = LinearToSrgb(ToneMap(hdr_image, 1.5))

        total_path_length_sum = np.sum(context['path_length_buffer'].to_array())
        scatter_type = context['scatter_type_buffer'].to_array()

        # total_brdf_scatter_sum = np.sum(scatter_type[:, :, 0])
        # total_q_scatter_sum = np.sum(scatter_type[:, :, 1])
        # total_q_scatter_rate = total_q_scatter_sum / (total_q_scatter_sum + total_brdf_scatter_sum)
        # total_q_scatter_img = scatter_type[:, :, 1] / (1e-6 + scatter_type[:, :, 0] + scatter_type[:, :, 1])
        # total_q_scatter_img = np.flipud(total_q_scatter_img)
        # print("Q scatter rate", total_q_scatter_rate)

        elapsed_time = end_time - start_time
        elapsed_time_optix_core = float(np.sum(np.asarray(elapsed_times)))
        elapsed_time_per_sample = elapsed_time / completed_samples
        print("Elapsed Time:", str(timedelta(seconds=elapsed_time)))
        print("Elapsed Time OptiX Core:", str(timedelta(seconds=elapsed_time_optix_core)))
        print("Accumulated time", str(timedelta(seconds=q_table_update_elapsed_time_accumulated)))

        error_mean = 0.1
        if self.reference_image is not None and self.reference_image.shape == ldr_image.shape:
            error = np.abs(ldr_image - self.reference_image)
            error_mean = np.mean(error)

        np_hit_count = context['hit_count_buffer'].to_array()
        total_hit_count = np.sum(np_hit_count)
        total_hit_percentage = total_hit_count / ((width * height) * completed_samples)
        print("Hit percent", total_hit_percentage)

        if show_picture:
            plt.imshow(ldr_image)
            plt.show()
            #plt.imshow(total_q_scatter_img)
            #plt.show()

        results = dict()

        # image
        results["image"] = ldr_image

        # sequence
        results["hit_count_sequence"] = hit_counts
        results["elapsed_times"] = elapsed_times_single_sample[time_limit_init_ignore_step:-1]

        # float
        results["elapsed_time_per_sample_except_init"] = np.mean(np.array(elapsed_times_single_sample[time_limit_init_ignore_step:-1]))
        results["elapsed_time_per_sample"] = elapsed_time_per_sample
        results["error_mean"] = error_mean
        results["total_hit_percentage"] = total_hit_percentage
        results["total_elapsed_time"] = elapsed_time
        results["completed_samples"] = completed_samples

        if need_q_table_update:
            results["q_table_info"] = q_table

        return results
