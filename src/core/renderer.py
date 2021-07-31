from pyoptix import Context, Program, Material
from core.scene import Scene
from core.renderer_utils import *
import time
from core.utils.math_utils import *
import matplotlib.pyplot as plt

from path_guiding.quadtree import *
from core.renderer_constants import *
from path_guiding.radiance_record import QTable
from utils.logging_utils import *
from utils.timing_utils import *
import gc
from core.optix_scene import OptiXSceneContext


class Renderer:
    def __init__(self, scale=1, force_all_diffuse=False):
        """
        This is created only once!
        :param scale:
        :param force_all_diffuse:
        """
        # Optix Context
        self.optix_context = None

        self.width = 0
        self.height = 0

        self.scene = None
        self.scene_name = None
        self.scale = scale
        self.reference_image = None
        self.scene_octree = None
        self.context = None

        self.render_load_logger = load_logger('Render load logger')
        self.render_logger = load_logger('Render logger')
        self.render_logger.setLevel(logging.INFO)

    def init_scene_config(self, scene_name):
        # load scene info (non optix)
        self.scene = Scene(scene_name)
        self.scene_name = scene_name
        self.scene.load_scene_from("../scene/%s/scene.xml" % scene_name)
        self.width = self.scene.width // self.scale
        self.height = self.scene.height // self.scale

    def reset_output_buffers(self, width, height):
        self.context['hit_count_buffer'] = Buffer.empty((height, width, 1), dtype=np.uint32, buffer_type='o', drop_last_dim=True)
        self.context['path_length_buffer'] = Buffer.empty((height, width, 1), dtype=np.uint32, buffer_type='o', drop_last_dim=True)
        self.context['scatter_type_buffer'] = Buffer.empty((height, width, 2), dtype=np.float32, buffer_type='o', drop_last_dim=True)
        self.context['output_buffer'] = Buffer.empty((height, width, 4), dtype=np.float32, buffer_type='o', drop_last_dim=True)
        self.context['output_buffer2'] = Buffer.empty((height, width, 4), dtype=np.float32, buffer_type='o',  drop_last_dim=True)

    def load_scene(self, scene_name):
        if self.scene_name != scene_name:
            del self.optix_context
            del self.scene
            gc.collect()
            self.context = Context()

            with time_measure("[1] Optix Context Create", self.render_load_logger):
                self.optix_context = OptiXSceneContext(self.context)

            with time_measure("[2] Scene Config Load", self.render_load_logger):
                self.init_scene_config(scene_name)

            with time_measure("[3] OptiX Load", self.render_load_logger):
                self.optix_context.load_scene(self.scene)

        else:
            self.render_load_logger.info("Skipped loading scene because it has been already loaded")

    def render(
            self,
            scene_name="cornell-box",
            spp=256,
            time_limit_in_sec=-1,
            time_limit_init_ignore_step=0,
            time_consider_only_optix=False,
            sampling_strategy=SAMPLE_COSINE,
            q_table_update_method=Q_UPDATE_MONTE_CARLO,
            show_picture=False,
            samples_per_pass=16,
            learning_method="incremental",
            accumulative_q_table_update=True,
            max_depth=8,
            rr_begin_depth=4,
            uv_n=8,
            n_cube=32,
            spatial_type='grid',
            directional_type='grid',
            directional_mapping_method="equal_area",
            use_bsdf_first_force=True,
            force_update_q_table=False,
            bsdf_sampling_fraction=0.0,
            min_epsilon=0.0,
            no_exploration=False,
            convert_ldr=True,
            quad_tree_update_type='gpu',
            **kwargs
    ):
        # load scene info & init optix
        self.load_scene(scene_name)

        # just for shorter name
        context = self.context
        scene = self.scene
        width = self.width
        height = self.height

        # rendering related
        context['rr_begin_depth'] = np.array(rr_begin_depth, dtype=np.uint32)
        context['max_depth'] = np.array(max_depth, dtype=np.uint32)
        context["bsdf_sampling_fraction"] = np.array(bsdf_sampling_fraction, dtype=np.float32)
        context["sampling_strategy"] = np.array(sampling_strategy, dtype=np.uint32)
        context["q_table_update_method"] = np.array(q_table_update_method, dtype=np.uint32)
        context["accumulative_q_table_update"] = np.array(1 if accumulative_q_table_update else 0, dtype=np.uint32)

        need_q_table_update = force_update_q_table or not ((sampling_strategy == SAMPLE_UNIFORM) or (sampling_strategy == SAMPLE_COSINE))

        context["need_q_table_update"] = np.array(1 if need_q_table_update else 0, dtype=np.uint32)

        room_size = scene.bbox.bbox_max - scene.bbox.bbox_min
        context['scene_bbox_min'] = scene.bbox.bbox_min
        context['scene_bbox_max'] = scene.bbox.bbox_max
        context['scene_bbox_extent'] = room_size

        self.reset_output_buffers(width, height)
        QTable.register_empty_context(context)

        q_table = None
        if need_q_table_update:
            q_table = QTable(
                spatial_type=spatial_type,
                directional_type=directional_type,
                directional_mapping_method=directional_mapping_method,
                accumulative_q_table_update=accumulative_q_table_update,
                n_cube=n_cube, n_uv=uv_n, octree=self.scene_octree,
                quad_tree_update_type=quad_tree_update_type
            )
            q_table.register_to_context(context)

        current_samples_per_pass = samples_per_pass
        if samples_per_pass == -1:
            current_samples_per_pass = spp

        hit_sum = 0
        list_hit_counts = []
        output_images = []

        is_budget_time = time_limit_in_sec > 0

        context.validate()
        context.compile()

        list_count_sample_per_pass = []
        list_time_optix_launch = []
        list_time_q_table_update = []
        need_to_save_intermediate_outputs = False

        left_samples = spp
        completed_samples = 0
        n_pass = 0

        counter_sample_next_q_table_update = 0
        counter_sample_exponential_update_size = 4
        counter_q_table_update = 0

        main_render_loop_start_time = time.time()
        '''
        Main Render Loop
        '''
        try:
            with timeout(time_limit_in_sec):
                while left_samples > 0:
                    list_count_sample_per_pass.append(current_samples_per_pass)
                    context["samples_per_pass"] = np.array(current_samples_per_pass, dtype=np.uint32)
                    self.render_logger.debug(
                        "Current Pass: %d, Current Samples: %d" % (n_pass, current_samples_per_pass))

                    if (not no_exploration) and sampling_strategy != SAMPLE_Q_SPHERE:
                        epsilon = getEpsilon(completed_samples, 100000, t=1, k=100)
                        epsilon = max(epsilon, min_epsilon)
                    else:
                        epsilon = 0.0

                    context["completed_sample_number"] = np.array(completed_samples, dtype=np.uint32)
                    if n_pass == 0 and use_bsdf_first_force:
                        context['sampling_strategy'] = np.array(SAMPLE_COSINE, dtype=np.uint32)
                    else:
                        context['sampling_strategy'] = np.array(sampling_strategy, dtype=np.uint32)
                        context['bsdf_sampling_fraction'] = np.array(bsdf_sampling_fraction, dtype=np.float32)

                    # Run OptiX program
                    with record_elapsed_time("OptiX Launch", list_time_optix_launch, self.render_logger):
                        context.launch(0, width, height)

                    # Update/refine radiance record if needed
                    if need_q_table_update and completed_samples >= counter_sample_next_q_table_update:
                        with record_elapsed_time("Q table update time", list_time_q_table_update, self.render_logger):
                            q_table.update_pdf(context, epsilon, sampling_strategy is SAMPLE_Q_SPHERE,
                                               quad_tree_update_type, k=counter_q_table_update, **kwargs)
                        if learning_method == "exponential":
                            counter_sample_exponential_update_size *= 2
                            counter_sample_next_q_table_update += counter_sample_exponential_update_size
                        else:
                            counter_sample_next_q_table_update += samples_per_pass
                        counter_q_table_update += 1

                    np_hit_count = context['hit_count_buffer'].to_array()
                    hit_new_sum = np.sum(np_hit_count)

                    if n_pass > 0:
                        hit = hit_new_sum - hit_sum
                        list_hit_counts.append(hit)

                    hit_sum = hit_new_sum

                    completed_samples += current_samples_per_pass

                    if need_to_save_intermediate_outputs:
                        output_image = self.context['output_buffer'].to_array()
                        output_images.append(output_image / completed_samples)

                    if not is_budget_time:
                        # update next pass
                        left_samples -= current_samples_per_pass
                        current_samples_per_pass = samples_per_pass
                        current_samples_per_pass = min(current_samples_per_pass, left_samples)
                    n_pass += 1
        except TimeoutError:
            self.render_logger.info("%f sec is over" % time_limit_in_sec)

        main_render_loop_end_time = time.time()
        total_render_elapsed_time = main_render_loop_end_time - main_render_loop_start_time

        # (0) Image
        final_raw_image = self.context['output_buffer'].to_array()
        final_raw_image = final_raw_image / completed_samples
        final_raw_image = np.flipud(final_raw_image)
        hdr_image = final_raw_image[:, :, 0:3]
        ldr_image = LinearToSrgb(ToneMap(hdr_image, 1.5))
        final_image = ldr_image if convert_ldr else hdr_image

        if show_picture:
            plt.imshow(final_image)
            plt.show()

        # (1) Error with reference image
        error_mean = 0.1
        if self.reference_image is not None and self.reference_image.shape == final_image.shape:
            error = np.abs(final_image - self.reference_image)
            error_mean = np.mean(error)

        # (2) Hit count sequences (how many paths reached the light source?)
        np_hit_count = context['hit_count_buffer'].to_array()
        total_hit_count = np.sum(np_hit_count)
        total_hit_percentage = total_hit_count / ((width * height) * completed_samples)
        list_hit_rates = [t / (n * width * height) for t, n in zip(list_hit_counts, list_count_sample_per_pass)]

        # (3) execution time
        list_time_optix_launch_per_sample = [t / n for t, n in zip(list_time_optix_launch,
                                                                             list_count_sample_per_pass)]
        average_time_optix_launch_per_sample = \
            np.mean(np.asarray(list_time_optix_launch_per_sample))
        average_time_optix_launch_per_sample_except_init = \
            np.mean(np.asarray(list_time_optix_launch_per_sample[1:-1]))

        if len(list_time_q_table_update) == 0:
            list_time_q_table_update.append(0)

        # Summarize Result

        results = dict()

        # image
        results["image"] = final_image

        # sequence
        results["hit_rate_per_pass"] = list_hit_rates
        results["elapsed_time_per_sample_per_pass"] = list_time_optix_launch_per_sample[1:-1]
        results["q_table_update_times"] = list_time_q_table_update

        # scalar
        # (1) Time
        #  - per sample
        results["elapsed_time_per_sample_except_init"] = average_time_optix_launch_per_sample_except_init
        results["elapsed_time_per_sample"] = average_time_optix_launch_per_sample
        #  - total
        results["total_elapsed_time"] = total_render_elapsed_time
        results["total_q_table_update_time"] = np.sum(np.asarray(list_time_q_table_update))
        results["total_optix_launch_time"] = np.sum(np.asarray(list_time_optix_launch))

        # (2) others
        results["error_mean"] = error_mean
        results["total_hit_percentage"] = total_hit_percentage
        results["completed_samples"] = completed_samples

        if need_q_table_update:
            results["q_table_info"] = q_table
            results["invalid_sample_rate"] = q_table.get_invalid_sample_rate(context)
        else:
            results["invalid_sample_rate"] = 0

        self.render_logger.info("[Rendering complete]")
        for key, v in results.items():
            self.render_logger.info("\t -%s : %s" % (str(key), str(v)))

        return results
