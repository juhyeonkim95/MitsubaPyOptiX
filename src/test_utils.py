from core.renderer import *
import datetime
from utils.image_utils import *
from utils.result_export_utils import *

from collections import OrderedDict
import pandas as pd
import json
from utils.result_visualize_utils import *


def show_radiance_map(scene_name, scale=4):
    reference_parent_folder = '../reference_images/%s/scale_%d' % ("standard", scale)
    ref_image = load_reference_image(reference_parent_folder, scene_name)

    renderer = Renderer(scale=scale, force_all_diffuse=False)
    renderer.reference_image = ref_image

    n_cube = 8
    coord = (2, 7, 4)
    common_params = {'scene_name': scene_name, 'samples_per_pass': 8, 'show_picture': True, 'max_depth': 16,
                     'rr_begin_depth': 8, 'scene_epsilon': 1e-5, 'accumulative_q_table_update': True,
                     'n_cube': n_cube, '_spp': 1024, 'time_limit_init_ignore_step': 0}

    pos = np.array(coord, dtype=np.float32) / n_cube
    size = np.array([1 / n_cube] * 3, dtype=np.float32)
    renderer.render(**common_params, spherical_pos=pos, spherical_size=size, spherical_map_type=1, convert_ldr=False)


def test_single_scene(scene_name,
                      scale=4,
                      test_time=False,
                      show_picture=False,
                      show_result=False,
                      output_folder=None,
                      _time=5,
                      _spp=256,
                      _sample_type=SAMPLE_Q_COS_REJECT,
                      _update_type=Q_UPDATE_MONTE_CARLO,
                      test_target=1,
                      do_bsdf=True,
                      visualize_octree=False):
    try:
        reference_parent_folder = '../reference_images/%s/scale_%d' % ("standard", scale)
        ref_image = load_reference_image(reference_parent_folder, scene_name)
    except Exception as e:
        ref_image = None

    total_results = OrderedDict()
    renderer = Renderer(scale=scale)
    renderer.reference_image = ref_image

    common_params = {
        'scene_name': scene_name,
        'samples_per_pass': 16,
        'show_picture': show_picture,
        'max_depth': 8,
        'rr_begin_depth': 8,
        'scene_epsilon': 1e-5,
        # You should change q_table_old at getQValue to q_table
        'accumulative_q_table_update': True,
        'n_uv': 16,
        'n_cube': 16,
        'use_mis': False,
        'clear_accumulated_info_per_update': False
    }

    if test_time:
        common_params['time_limit_in_sec'] = _time
    else:
        common_params['spp'] = _spp
    common_params['time_limit_init_ignore_step'] = 10

    if do_bsdf:
        total_results["brdf"] = renderer.render(**common_params, sampling_strategy=SAMPLE_BRDF)

    def test_2():
        common_params2 = {
            "sampling_strategy": SAMPLE_MIS,
            "directional_mapping_method": "cylindrical",
            "directional_data_structure_type": "quadtree",
            "learning_method": "exponential",
            "bsdf_sampling_fraction": 0.5,
            "q_table_update_method": Q_UPDATE_MONTE_CARLO,
            "binary_tree_split_sample_number": 12000
        }

        common_params_total = {**common_params, **common_params2}

        #common_params_total['spatial_data_structure_type'] = 'binary_tree'
        #common_params_total['directional_data_structure_type'] = 'quadtree'
        #common_params_total["directional_mapping_method"] = "cylindrical"

        common_params_total['spatial_data_structure_type'] = 'grid'
        common_params_total['directional_data_structure_type'] = 'grid'
        common_params_total["directional_mapping_method"] = "cylindrical"
        common_params_total["learning_method"] = "incremental"

        total_results["binary_tree_quadtree_cpu_single"] = renderer.render(**common_params_total, quad_tree_update_type='cpu_single')
        common_params_total["q_table_update_method"] = Q_UPDATE_SARSA
        total_results["binary_tree_quadtree_cpu_single_sarsa"] = renderer.render(**common_params_total, quad_tree_update_type='cpu_single')

        common_params_total["q_table_update_method"] = Q_UPDATE_MONTE_CARLO
        total_results["binary_tree_quadtree_cpu_single"] = renderer.render(**common_params_total, quad_tree_update_type='cpu_single')
        common_params_total["q_table_update_method"] = Q_UPDATE_SARSA
        total_results["binary_tree_quadtree_cpu_single_sarsa"] = renderer.render(**common_params_total, quad_tree_update_type='cpu_single')

    def test_compare_all_grid():
        common_params2 = {
            "directional_mapping_method": "cylindrical",
            "directional_data_structure_type": "grid",
            "spatial_data_structure_type": "grid",
            "learning_method": "linear",
            "bsdf_sampling_fraction": 0.5
        }
        common_params_total = {**common_params, **common_params2}

        sampling_strategies = ["mis", "qcos_inversion", "reject_mix"]
        q_table_update_methods = ["mc", "sarsa"]

        for sampling_strategy in sampling_strategies:
            for q_table_update_method in q_table_update_methods:
                common_params_total["sampling_strategy"] = key_value_to_int("sampling_strategy", sampling_strategy)
                common_params_total["q_table_update_method"] = key_value_to_int("q_table_update_method", q_table_update_method)
                total_results["%s_%s" % (sampling_strategy, q_table_update_method)] = renderer.render(**common_params_total)

    def test_3():
        common_params2 = {
            "sampling_strategy": SAMPLE_MIS,
            "directional_mapping_method": "cylindrical",
            "directional_data_structure_type": "quadtree",
            "learning_method": "exponential",
            "bsdf_sampling_fraction": 0.5,
            "q_table_update_method": Q_UPDATE_MONTE_CARLO,
            "binary_tree_split_sample_number": 12000
        }

        common_params_total = {**common_params, **common_params2}

        common_params_total['spatial_data_structure_type'] = 'grid'
        common_params_total['directional_data_structure_type'] = 'grid'
        common_params_total["directional_mapping_method"] = "shirley"
        total_results["grid_grid_cpu_single"] = renderer.render(**common_params_total, quad_tree_update_type='cpu_single')

        common_params_total['directional_data_structure_type'] = 'quadtree'
        common_params_total["directional_mapping_method"] = "cylindrical"
        total_results["grid_quadtree_cpu_single"] = renderer.render(**common_params_total, quad_tree_update_type='cpu_single')

        common_params_total['spatial_data_structure_type'] = 'binary_tree'
        common_params_total['directional_data_structure_type'] = 'grid'
        common_params_total["directional_mapping_method"] = "shirley"
        total_results["binary_tree_grid_cpu_single"] = renderer.render(**common_params_total, quad_tree_update_type='cpu_single')

        common_params_total['directional_data_structure_type'] = 'quadtree'
        common_params_total["directional_mapping_method"] = "cylindrical"
        total_results["binary_tree_quadtree_cpu_single"] = renderer.render(**common_params_total, quad_tree_update_type='cpu_single')

        common_params_total["q_table_update_method"] = Q_UPDATE_SARSA
        # common_params_total["learning_method"] = "linear"

        common_params_total['spatial_data_structure_type'] = 'grid'
        common_params_total['directional_data_structure_type'] = 'grid'
        common_params_total["directional_mapping_method"] = "shirley"
        total_results["grid_grid_cpu_single_sarsa"] = renderer.render(**common_params_total, quad_tree_update_type='cpu_single')

        common_params_total['directional_data_structure_type'] = 'quadtree'
        common_params_total["directional_mapping_method"] = "cylindrical"
        total_results["grid_quadtree_cpu_single_sarsa"] = renderer.render(**common_params_total, quad_tree_update_type='cpu_single')

        common_params_total['spatial_data_structure_type'] = 'binary_tree'
        common_params_total['directional_data_structure_type'] = 'grid'
        common_params_total["directional_mapping_method"] = "shirley"
        total_results["binary_tree_grid_cpu_single_sarsa"] = renderer.render(**common_params_total, quad_tree_update_type='cpu_single')

        common_params_total['directional_data_structure_type'] = 'quadtree'
        common_params_total["directional_mapping_method"] = "cylindrical"
        total_results["binary_tree_quadtree_cpu_single_sarsa"] = renderer.render(**common_params_total, quad_tree_update_type='cpu_single')


        #common_params_total['directional_data_structure_type'] = 'grid'
        #common_params_total["directional_mapping_method"] = "shirley"
        #total_results["grid_grid_cpu_single"] = renderer.render(**common_params_total, quad_tree_update_type='cpu_single')

        #total_results["binary_tree_cpu_multi"] = renderer.render(**common_params_total, quad_tree_update_type='cpu_multi')
        #total_results["binary_tree_gpu"] = renderer.render(**common_params_total, quad_tree_update_type='gpu')
        #common_params_total['spatial_data_structure_type'] = 'grid'
        #total_results["grid_cpu_single"] = renderer.render(**common_params_total, quad_tree_update_type='cpu_single')
        #total_results["grid_cpu_multi"] = renderer.render(**common_params_total, quad_tree_update_type='cpu_multi')
        #total_results["grid_gpu"] = renderer.render(**common_params_total, quad_tree_update_type='gpu')

    #if test_target == 2:
    #    test_2()

    test_compare_all_grid()

    if show_result:
        show_result_func(total_results)

    if output_folder is not None:
        scene_output_folder = "%s/%s" % (output_folder, scene_name)
        export_total_results(total_results, scene_output_folder)

    return total_results


def test_multiple_and_export_result(scene_list, scale, output_folder, _time=5, _spp=256, test_time=False, test_target=1):
    for scene in scene_list:
        #try:
        test_single_scene(scene, scale, test_time=test_time, show_picture=False, show_result=False, _time=_time,
                          _spp=_spp, output_folder=output_folder, test_target=test_target)
        # except Exception:
        #     print("Scene Error")

    update_total_result(output_folder)


def test_octree_build(scene_name, scale=2, visualize=False):
    #test_single_scene(scene_name, test_target=6)
    renderer = Renderer(scale=scale)
    renderer.construct_stree(scene_name, max_octree_depth=6, visualize=visualize)
