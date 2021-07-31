from pyoptix import Context, Program, Material
from core.scene import Scene

import time
from core.utils.math_utils import *
import matplotlib.pyplot as plt
from datetime import timedelta

from path_guiding.quadtree import *
from core.renderer_constants import *
from path_guiding.radiance_record import QTable
#from path_guiding.qTable import QTable
from utils.logging_utils import *
from utils.timing_utils import timing
import gc


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
        self.force_all_diffuse = force_all_diffuse
        self.reference_image = None
        self.scene_octree = None

        self.compile_config = {}

    @timing
    def init_optix_context(self):
        if self.context is None:
            context = Context()
        else:
            context = self.context

        context.set_ray_type_count(2)
        context.set_entry_point_count(2)
        context.set_stack_size(8000)
        # context.set_devices([0,1,2,3,4])
        # print("Device Name", context.get_device_name(0))

        context['scene_epsilon'] = np.array(self.scene_epsilon, dtype=np.float32)
        context['pathtrace_ray_type'] = np.array(0, dtype=np.uint32)
        context['pathtrace_shadow_ray_type'] = np.array(1, dtype=np.uint32)
        context['bad_color'] = np.array([1000000., 0., 1000000.], dtype=np.float32)
        context['bg_color'] = np.zeros(3, dtype=np.float32)
        self.context = context

        #context.set_print_enabled(True)
        #context.set_exception_enabled(ExceptionType.all, True)
        #context.set_usage_report_callback(1)

        # print("get_cpu_num_of_threads", context.get_cpu_num_of_threads())
        # print("get_stack_size", context.get_stack_size())
        # print("get_available_devices_count", context.get_available_devices_count())
        # print("get_enabled_device_count", context.get_enabled_device_count())
        # print("get_used_host_memory", context.get_used_host_memory())
        # print("get_available_device_memory", context.get_available_device_memory(0))

    @timing
    def init_entry_point(self, has_envmap):
        if has_envmap:
            miss_program = self.program_dictionary["miss_envmap"]
        else:
            miss_program = self.program_dictionary["miss"]

        self.context.set_ray_generation_program(0, self.program_dictionary['ray_generation'])
        self.context.set_exception_program(0, self.program_dictionary['exception'])
        self.context.set_miss_program(0, miss_program)

        self.context.set_ray_generation_program(1, self.program_dictionary["quad_tree_updater"])
        # self.context.set_ray_generation_program(2, self.program_dictionary["binary_tree_updater"])

        # self.context.set_ray_generation_program(1, self.program_dictionary['quad_tree_updater'])
        # self.entry_point = EntryPoint(self.program_dictionary['ray_generation'],
        #                          self.program_dictionary['exception'],
        #                          miss_program)
        # self.quad_tree_updater_entry_point = EntryPoint()

    def update_compile_config(self, use_nee, sampling_strategy):
        updated = False
        if "use_nee" not in self.compile_config or self.compile_config["use_nee"] != use_nee:
            updated = True
            self.compile_config["use_nee"] = use_nee
        if "sampling_strategy" not in self.compile_config or self.compile_config["sampling_strategy"] != sampling_strategy:
            updated = True
            self.compile_config["sampling_strategy"] = sampling_strategy
        if updated:
            self.edit_compile_config(use_nee, sampling_strategy)
        return updated

    def edit_compile_config(self, use_nee, sampling_strategy):
        with open("optix/app_config.h", 'w') as f:
            lines = [
                "#ifndef APP_CONFIG_H",
                "#define APP_CONFIG_H",
                "#define USE_NEXT_EVENT_ESTIMATION %d" % (1 if use_nee else 0),
                "",
                "#define SAMPLING_STRATEGY_BSDF %d" % SAMPLE_COSINE,
                "#define SAMPLING_STRATEGY_SD_TREE %d" % SAMPLE_Q_QUADTREE,
                "#define SAMPLING_STRATEGY %d" % sampling_strategy,
                "#endif"
            ]
            f.write('\n'.join(lines))

    @timing
    def init_optix_programs(self):
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
        program_dictionary["box_bb"] = Program('optix/shapes/box.cu', 'box_bounds')
        program_dictionary["box_it"] = Program('optix/shapes/box.cu', 'box_intersect')

        # Materials
        closest_hit_program = 'optix/integrators/hit_program.cu'
        any_hit_hit_program = 'optix/integrators/any_hit_program.cu'

        program_dictionary["closest_hit"] = Program(closest_hit_program, 'closest_hit')
        program_dictionary["closest_hit_light"] = Program('optix/integrators/light_hit_program.cu', 'diffuseEmitter')
        program_dictionary["any_hit_cutout"] = Program(any_hit_hit_program, 'any_hit_cutout')

        program_dictionary["any_hit_shadow"] = Program(any_hit_hit_program, 'any_hit_shadow')
        program_dictionary["any_hit_shadow_cutout"] = Program(any_hit_hit_program, 'any_hit_shadow_cutout')

        program_dictionary['quad_tree_updater'] = Program('optix/q_table/quad_tree_updater.cu', 'quad_tree_updater')
        program_dictionary['binary_tree_updater'] = Program('optix/q_table/binary_tree_updater.cu', 'spatial_binary_tree_updater')

        self.program_dictionary = program_dictionary

    @timing
    def init_material_program_dict(self):
        material_dict = {}
        program_dictionary = self.program_dictionary

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

    @timing
    def init_optix_scene(self):
        """
        Init Optix related things to the scene
        :return:
        """
        scene = self.scene
        context = self.context

        # (1) load texture data to optix and retrieve optix texture ids
        scene.optix_load_textures()
        # (2) load OBJ mesh data to optix and retrieve optix buffer ids
        scene.optix_create_objs(self.program_dictionary)
        # (3) from (1) and (2), create optix geometry instances
        scene.optix_create_geometry_instances(self.program_dictionary, self.material_dict, self.force_all_diffuse)

        # Assign optix objects to context
        create_geometry(context, scene)
        create_scene_lights(context, scene)
        create_scene_materials(context, scene)

    def init_spherical_camera(self, pos, size, map_type):
        room_size = self.scene.bbox.bbox_max - self.scene.bbox.bbox_min
        self.context["spherical_cam_position"] = pos * room_size + self.scene.bbox.bbox_min
        self.context["spherical_cam_size"] = size * room_size
        self.context["spherical_cam_directional_mapping"] = np.array(map_type, dtype=np.uint32)
        self.context["camera_type"] = np.array(1, dtype=np.uint32)

    @timing
    def init_camera(self):
        aspect_ratio = float(self.width) / float(self.height)
        camera = self.scene.camera
        u, v, w = camera.calc_image_space_vectors(aspect_ratio)

        # upload to context
        context = self.context
        context["eye"] = np.array(camera.eye, dtype=np.float32)
        context["U"] = np.array(u, dtype=np.float32)
        context["V"] = np.array(v, dtype=np.float32)
        context["W"] = np.array(w, dtype=np.float32)
        context["focalDistance"] = np.array(5, dtype=np.float32)
        # context["apertureRadius"] = np.array(0.5, dtype=np.float32)
        context["camera_type"] = np.array(0, dtype=np.uint32)

    def init_scene_config(self, scene_name):
        # load scene info (non optix)
        self.scene = Scene(scene_name)
        self.scene.load_scene_from("../scene/%s/scene.xml" % scene_name)
        self.width = self.scene.width // self.scale
        self.height = self.scene.height // self.scale

    @timing
    def init_optix(self):
        # init general optix info
        self.init_optix_context()
        self.init_optix_programs()
        self.init_entry_point(self.scene.has_envmap)

        # init scene specific information
        self.init_material_program_dict()
        self.init_optix_scene()
        self.init_camera()

    def register_context_related(self):
        context = self.context
        height = self.height
        width = self.width
        context['hit_count_buffer'] = Buffer.empty((height, width, 1), dtype=np.float32, buffer_type='o', drop_last_dim=True)
        context['path_length_buffer'] = Buffer.empty((height, width, 1), dtype=np.float32, buffer_type='o', drop_last_dim=True)
        context['scatter_type_buffer'] = Buffer.empty((height, width, 2), dtype=np.float32, buffer_type='o', drop_last_dim=True)
        context['output_buffer'] = Buffer.empty((height, width, 4), dtype=np.float32, buffer_type='o', drop_last_dim=True)
        context['output_buffer2'] = Buffer.empty((height, width, 4), dtype=np.float32, buffer_type='o',  drop_last_dim=True)

    def render(
            self,
            scene_name="cornell-box",
            spp=256,
            time_limit_in_sec=-1,
            time_limit_init_ignore_step=0,
            time_consider_only_optix=False,
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
            spherical_map_type=0,
            bsdf_sampling_fraction=0.0,
            min_epsilon=0.0,
            no_exploration=False,
            convert_ldr=True,
            use_memoization=True,
            quad_tree_update_type='gpu',
            path_guiding_configs=None,
            **kwargs
    ):
        self.scene_epsilon = 1e-3 if scene_name == "veach_door_simple" else 1e-5

        is_budget_time = (time_limit_in_sec > 0)
        start_time = time.time()

        #config_updated = self.update_compile_config(use_nee=use_mis, sampling_strategy=sampling_strategy)

        if self.scene_name != scene_name:
            del self.context
            gc.collect()
            self.context = None
            self.scene_name = scene_name
            self.init_scene_config(scene_name)
            self.init_optix()
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

        context["bsdf_sampling_fraction"] = np.array(bsdf_sampling_fraction, dtype=np.float32)
        context["sigma_s"] = np.array(0, dtype=np.float32)
        context["sigma_a"] = np.array(0, dtype=np.float32)
        context["sigma_t"] = np.array(0, dtype=np.float32)
        context["hg_g"] = np.array(0, dtype=np.float32)

        context['rr_begin_depth'] = np.array(rr_begin_depth, dtype=np.uint32)
        context['max_depth'] = np.array(max_depth, dtype=np.uint32)
        context["sampling_strategy"] = np.array(sample_type, dtype=np.uint32)
        context["use_mis"] = np.array(1 if use_mis and not scene.has_envmap else 0, dtype=np.uint32)
        context["use_tone_mapping"] = np.array(1 if use_tone_mapping else 0, dtype=np.uint32)
        context["use_soft_q_update"] = np.array(1 if use_soft_q_update else 0, dtype=np.uint32)
        context["save_q_cos"] = np.array(1 if save_q_cos else 0, dtype=np.uint32)
        context["use_memoization"] = np.array(1 if use_memoization else 0, dtype=np.uint32)

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
            q_table = QTable(
                spatial_type=spatial_type,
                directional_type=directional_type,
                directional_mapping_method=directional_mapping_method,
                accumulative_q_table_update=accumulative_q_table_update,
                n_cube=n_cube, n_uv=uv_n, octree=self.scene_octree
            )
            q_table.register_to_context(context)

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
        need_to_save_intermediate_outputs = sample_combination == "inverse_variance" or sample_combination == "discard"

        final_image = np.zeros((height, width, 4))
        q_table_update_elapsed_time_accumulated = 0

        is_first = True

        print("Optix Prepare Time:", str(timedelta(seconds=time.time() - start_time)))

        start_time = time.time()
        start_time_2 = None
        elapsed_time_optix = 0
        exponential_update_counter = 0
        q_table_update_counter = 0
        exponential_update_size = 4
        if q_table:
            zeros_n_s_n_a = np.zeros((q_table.n_s, q_table.n_a), dtype=np.float32)

        # from core.utils.windows_utils import ImageWindowBase
        # window = ImageWindowBase(context, width, height)
        # window.run()

        while True:
            if is_budget_time:
                if n_pass > time_limit_init_ignore_step:
                    if time_consider_only_optix:
                        if elapsed_time_optix > time_limit_in_sec:
                            break
                    else:
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

            if (not no_exploration) and sample_type != SAMPLE_Q_SPHERE:
                epsilon = getEpsilon(completed_samples, 100000, t=1, k=100)
                epsilon = max(epsilon, min_epsilon)

                print("Epsilon:", epsilon, "at", completed_samples)
            else:
                epsilon = 0.0

            context["frame_number"] = np.array((completed_samples + 1), dtype=np.uint32)



            # Run OptiX program
            ith_start_time = time.time()
            if is_first:
                if use_brdf_first_force:
                    context['bsdf_sampling_fraction'] = np.array(1, dtype=np.float32)
                else:
                    context['bsdf_sampling_fraction'] = np.array(0, dtype=np.float32)
                # self.entry_point.launch((width, height, b), context)
                context.validate()
                context.compile()
                context.launch(0, width, height, b)
                is_first = False
            else:
                context['bsdf_sampling_fraction'] = np.array(bsdf_sampling_fraction, dtype=np.float32)
                context.launch(0, width, height, b)

            # context.launch(1, width, height)

            ith_end_time = time.time()
            ith_time = (ith_end_time - ith_start_time)
            if n_pass >= time_limit_init_ignore_step:
                elapsed_time_optix += ith_time

            ith_time_single_sample = ith_time / current_samples_per_pass
            print("Ith _time", ith_time_single_sample, ith_time, b, current_samples_per_pass)
            elapsed_times.append(ith_time)
            elapsed_times_single_sample.append(ith_time_single_sample)

            def update_q_table():
                q_table_update_start_time = time.time()
                q_table.update_pdf(context, epsilon, sample_type is SAMPLE_Q_SPHERE,
                                   quad_tree_update_type, k=q_table_update_counter, **kwargs)

                context['irradiance_table'].copy_from_array(zeros_n_s_n_a)
                context['max_radiance_table'].copy_from_array(zeros_n_s_n_a)

                q_table_update_elapsed_time = time.time() - q_table_update_start_time
                return q_table_update_elapsed_time

            if need_q_table_update:
                if learning_method == 'exponential':
                    exponential_update_counter += current_samples_per_pass
                    if exponential_update_counter >= exponential_update_size:
                        print("Exponential update occurred! Size:", exponential_update_size)
                        q_table_update_elapsed_time_accumulated += update_q_table()
                        exponential_update_counter = 0
                        exponential_update_size *= 2
                        q_table_update_counter += 1
                else:
                    q_table_update_elapsed_time_accumulated += update_q_table()
                    q_table_update_counter += 1

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
                # if learning_method == "exponential":
                #     current_samples_per_pass *= 2
                #     if left_samples - current_samples_per_pass < 2 * current_samples_per_pass:
                #         current_samples_per_pass = left_samples
                # else:
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
        elif sample_combination == "discard":
            pass
        else:
            output_image = context['output_buffer'].to_array()
            final_image = output_image / completed_samples

        print(final_image.shape)
        if convert_ldr:
            final_image = np.flipud(final_image)

        hdr_image = final_image[:, :, 0:3]

        print(np.mean(hdr_image), "Image MEAN")

        # hdr_image = cv.cvtColor(hdr_image, cv.COLOR_RGB2BGR)
        # tone_map = cv.createTonemapReinhard(gamma=2.2, intensity=0, light_adapt=0, color_adapt=0)
        # ldr_image = tone_map.process(hdr_image)
        # ldr_image = cv.cvtColor(ldr_image, cv.COLOR_RGB2BGR)
        if convert_ldr:
            ldr_image = LinearToSrgb(ToneMap(hdr_image, 1.5))
        else:
            ldr_image = hdr_image
        A = np.max(ldr_image)
        B = np.min(ldr_image)

        print("Image max", A)
        print("Image min", B)

        elapsed_time = end_time - start_time
        elapsed_time_optix_core = float(np.sum(np.asarray(elapsed_times)))
        elapsed_time_per_sample = elapsed_time / completed_samples
        print("Elapsed Time:", str(timedelta(seconds=elapsed_time)))
        print("Elapsed Time OptiX Core:", str(timedelta(seconds=elapsed_time_optix_core)))
        print("Accumulated _time", str(timedelta(seconds=q_table_update_elapsed_time_accumulated)))

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
            results["invalid_sample_rate"] = q_table.get_invalid_sample_rate(context)
        else:
            results["invalid_sample_rate"] = 0

        return results
