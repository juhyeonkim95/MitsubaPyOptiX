from core.renderer import *
import datetime
from utils.image_utils import *
from utils.io_utils import *

from collections import OrderedDict
import pandas as pd
import json


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

    reference_parent_folder = '../reference_images/%s/scale_%d' % ("standard", scale)
    ref_image = load_reference_image(reference_parent_folder, scene_name)

    total_results = OrderedDict()
    renderer = Renderer(scale=scale)
    renderer.reference_image = ref_image

    common_params = {
        'scene_name': scene_name,
        'samples_per_pass': 8,
        'show_picture': show_picture,
        'max_depth': 16,
        'rr_begin_depth': 8,
        'scene_epsilon': 1e-5,
        # You should change q_table_old at getQValue to q_table
        'accumulative_q_table_update': True,
        'uv_n': 16,
        'n_cube': 8
    }

    if test_time:
        common_params['time_limit_in_sec'] = _time
    else:
        common_params['spp'] = _spp
    common_params['time_limit_init_ignore_step'] = 10
    # renderer.construct_stree(scene_name, max_octree_depth=4, max_path_depth=3)
    #
    # result = renderer.render(**common_params, sample_type=SAMPLE_COSINE, force_update_q_table=True)
    # normals = result['q_table_info'].q_table_normal_counts
    # normals = normals.astype(np.float32)
    # print(normals[122])
    # print(normals.dtype)
    # normals /= (np.sum(normals, axis=1, keepdims=True, dtype=np.float32) + 0.0001)
    # normals_stdev = np.std(normals, axis=1)
    # print(normals_stdev)
    # total_results["uniform"] = renderer.render(**common_params, sample_type=SAMPLE_UNIFORM)
    if do_bsdf:
        total_results["brdf"] = renderer.render(**common_params, sample_type=SAMPLE_COSINE)

    def test_1():
        # (1) Hemisphere - Inversions
        total_results["q_brdf_inv_sarsa"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_PROPORTION, q_table_update_method=Q_UPDATE_SARSA)
        total_results["q_brdf_inv_expected_sarsa"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_PROPORTION, q_table_update_method=Q_UPDATE_EXPECTED_SARSA)
        total_results["q_brdf_inv_mc"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_PROPORTION, q_table_update_method=Q_UPDATE_MONTE_CARLO)

        # (2) Hemisphere - Rejections
        total_results["q_brdf_rej_sarsa"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_REJECT, q_table_update_method=Q_UPDATE_SARSA)
        total_results["q_brdf_rej_expected_sarsa"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_REJECT, q_table_update_method=Q_UPDATE_EXPECTED_SARSA)
        total_results["q_brdf_rej_mc"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_REJECT, q_table_update_method=Q_UPDATE_MONTE_CARLO)

        # (2.5) Hemisphere - Rejections + MIX
        total_results["q_brdf_rej_sarsa_mix"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_REJECT_MIX, q_table_update_method=Q_UPDATE_SARSA)
        total_results["q_brdf_rej_expected_sarsa_mix"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_REJECT_MIX, q_table_update_method=Q_UPDATE_EXPECTED_SARSA)
        total_results["q_brdf_rej_mc_mix"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_REJECT_MIX, q_table_update_method=Q_UPDATE_MONTE_CARLO)

        # (3) Hemisphere - MCMC
        total_results["q_brdf_mcmc_sarsa"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_MCMC, q_table_update_method=Q_UPDATE_SARSA)
        total_results["q_brdf_mcmc_expected_sarsa"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_MCMC, q_table_update_method=Q_UPDATE_EXPECTED_SARSA)
        total_results["q_brdf_mcmc_mc"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_MCMC, q_table_update_method=Q_UPDATE_MONTE_CARLO)

        # (4) Sphere - Inversions (CDF used + 0.5 BRDF)
        total_results["q_mis_sphere_sarsa"] = renderer.render(**common_params, sample_type=SAMPLE_Q_SPHERE, q_table_update_method=Q_UPDATE_SARSA, bsdf_sampling_fraction=0.5)
        total_results["q_mis_sphere_expected_sarsa"] = renderer.render(**common_params, sample_type=SAMPLE_Q_SPHERE, q_table_update_method=Q_UPDATE_EXPECTED_SARSA, bsdf_sampling_fraction=0.5)
        total_results["q_mis_sphere_mc"] = renderer.render(**common_params, sample_type=SAMPLE_Q_SPHERE, q_table_update_method=Q_UPDATE_MONTE_CARLO, bsdf_sampling_fraction=0.5)

        # Muller 17
        total_results["q_mis_quadtree_mc_cpu_single"] = renderer.render(**common_params,
                                                                        sample_type=SAMPLE_Q_QUADTREE,
                                                                        quad_tree_update_type='cpu_single',
                                                                        force_update_q_table=True,
                                                                        directional_mapping_method="cylindrical",
                                                                        directional_type="quadtree",
                                                                        learning_method="exponential",
                                                                        bsdf_sampling_fraction=0.5,
                                                                        q_table_update_method=Q_UPDATE_MONTE_CARLO)
        total_results["q_mis_quadtree_mc_cpu_multi"] = renderer.render(**common_params,
                                                                       sample_type=SAMPLE_Q_QUADTREE,
                                                                       quad_tree_update_type='cpu_multi',
                                                                       force_update_q_table=True,
                                                                       directional_mapping_method="cylindrical",
                                                                       directional_type="quadtree",
                                                                       learning_method="exponential",
                                                                       bsdf_sampling_fraction=0.5,
                                                                       q_table_update_method=Q_UPDATE_MONTE_CARLO)
        total_results["q_mis_quadtree_mc_gpu"] = renderer.render(**common_params,
                                                                 sample_type=SAMPLE_Q_QUADTREE,
                                                                 quad_tree_update_type='gpu',
                                                                 force_update_q_table=True,
                                                                 directional_mapping_method="cylindrical",
                                                                 directional_type="quadtree",
                                                                 learning_method="exponential",
                                                                 bsdf_sampling_fraction=0.5,
                                                                 q_table_update_method=Q_UPDATE_MONTE_CARLO)

    def test_2():
        # common_params["time_consider_only_optix"] = True
        # total_results["q_brdf_rej_sarsa"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_REJECT,
        #                                                     q_table_update_method=Q_UPDATE_SARSA)

        # Muller 17
        # total_results["q_mis_quadtree_sarsa"] = renderer.render(**common_params,
        #                                                      sample_type=SAMPLE_Q_QUADTREE,
        #                                                      force_update_q_table=True,
        #                                                      directional_mapping_method="cylindrical",
        #                                                      directional_type="quadtree",
        #                                                      learning_method="exponential",
        #                                                      bsdf_sampling_fraction=0.5,
        #                                                      q_table_update_method=Q_UPDATE_SARSA)
        # Muller 17
        # common_params['spatial_type']='binary_tree'
        # total_results["q_mis_quadtree_mc_cpu_single"] = renderer.render(**common_params,
        #                                                      sample_type=SAMPLE_Q_QUADTREE,
        #                                                      quad_tree_update_type='cpu_single',
        #                                                      force_update_q_table=True,
        #                                                      directional_mapping_method="cylindrical",
        #                                                      directional_type="quadtree",
        #                                                      learning_method="exponential",
        #                                                      bsdf_sampling_fraction=0.5,
        #                                                      q_table_update_method=Q_UPDATE_MONTE_CARLO)
        # total_results["q_mis_quadtree_mc_cpu_multi"] = renderer.render(**common_params,
        #                                                      sample_type=SAMPLE_Q_QUADTREE,
        #                                                      quad_tree_update_type='cpu_multi',
        #                                                      force_update_q_table=True,
        #                                                      directional_mapping_method="cylindrical",
        #                                                      directional_type="quadtree",
        #                                                      learning_method="exponential",
        #                                                      bsdf_sampling_fraction=0.5,
        #                                                      q_table_update_method=Q_UPDATE_MONTE_CARLO)

        # total_results["grid"] = renderer.render(**common_params,
        #                                                      sample_type=SAMPLE_Q_QUADTREE,
        #                                                      quad_tree_update_type='gpu',
        #                                                      force_update_q_table=True,
        #                                                      directional_mapping_method="cylindrical",
        #                                                      directional_type="quadtree",
        #                                                      learning_method="exponential",
        #                                                      bsdf_sampling_fraction=0.5,
        #                                                      q_table_update_method=Q_UPDATE_MONTE_CARLO)
        # total_results["binary_tree"] = renderer.render(**common_params,
        #                                                          sample_type=SAMPLE_Q_QUADTREE,
        #                                                          quad_tree_update_type='gpu',
        #                                                          force_update_q_table=True,
        #                                                          directional_mapping_method="cylindrical",
        #                                                          directional_type="quadtree",
        #                                                          spatial_type="binary_tree",
        #                                                          learning_method="exponential",
        #                                                          bsdf_sampling_fraction=0.5,
        #                                                          q_table_update_method=Q_UPDATE_MONTE_CARLO,
        #                                                         binary_tree_split_sample_number=12000,
        #                                                         binary_tree_split_invalid_rate_threshold=1.0)
        add_params = {
            "sample_type":SAMPLE_Q_QUADTREE,
            "quad_tree_update_type": 'gpu',
            "directional_mapping_method": "cylindrical",
            "directional_type": "quadtree",
            "spatial_type": "binary_tree",
            "learning_method": "exponential",
            "bsdf_sampling_fraction" : 0.5,
            "q_table_update_method": Q_UPDATE_MONTE_CARLO
        }
        common_params2 = {**common_params, **add_params}
        #total_results["binary_tree_mc"] = renderer.render(**common_params2, q_table_update_method=Q_UPDATE_MONTE_CARLO)
        #total_results["binary_tree_expected_sarsa"] = renderer.render(**common_params2, q_table_update_method=Q_UPDATE_EXPECTED_SARSA)
        #total_results["binary_tree_sarsa"] = renderer.render(**common_params2, q_table_update_method=Q_UPDATE_SARSA)

        N = 10
        for i in range(N):
            total_results["binary_tree%d"%i] = renderer.render(**common_params2,
                                                                binary_tree_split_sample_number=12000,
                                                                binary_tree_split_invalid_rate_threshold=1.0/N*(i+1))

    def test_3(name, sample_type, update_type):
        #target_epsilon = [0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]
        target_epsilon = [0.0,
                          0.0001, 0.0002, 0.0005,
                          0.001, 0.002, 0.005,
                          0.01, 0.02, 0.05,
                          0.1, 0.2,
                          0.5,
                          0.8, 0.9,
                          0.95, 0.98, 0.99,
                          0.995, 0.998, 0.999,
                          1.0]
        for epsilon in target_epsilon:
            total_results[name+"_"+str(epsilon)] = renderer.render(**common_params, sample_type=sample_type,
                                                                   q_table_update_method=update_type, min_epsilon=epsilon)

    def test_4():
        total_results["q_brdf_rej_sarsa_mix"] = renderer.render(**common_params,
                                                                         sample_type=SAMPLE_Q_COS_REJECT_MIX,
                                                                         q_table_update_method=Q_UPDATE_SARSA,
                                                                        use_memoization=True)
        total_results["q_brdf_rej_sarsa_mix_no_memoization"] = renderer.render(**common_params,
                                                                         sample_type=SAMPLE_Q_COS_REJECT_MIX,
                                                                         q_table_update_method=Q_UPDATE_SARSA,
                                                                         use_memoization=False)

    def test_5():
        total_results["sarsa"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_REJECT,
                                                q_table_update_method=Q_UPDATE_SARSA)
        total_results["exp_sarsa"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_REJECT,
                                                q_table_update_method=Q_UPDATE_EXPECTED_SARSA)
        total_results["monte_carlo"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_REJECT,
                                                q_table_update_method=Q_UPDATE_MONTE_CARLO)

        # total_results["ours_no_explore"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_REJECT,
        #                                                         q_table_update_method=Q_UPDATE_SARSA, no_exploration=True)

        # Muller 17
        total_results["q_mis_quadtree_mc"] = renderer.render(**common_params,
                                                             sample_type=SAMPLE_Q_QUADTREE,
                                                             force_update_q_table=True,
                                                             directional_mapping_method="cylindrical",
                                                             directional_type="quadtree",
                                                             sample_combination="discard",
                                                             learning_method="exponential",
                                                             bsdf_sampling_fraction=0.5,
                                                             q_table_update_method=Q_UPDATE_MONTE_CARLO)

    def test_6():
        renderer.construct_stree(scene_name, max_octree_depth=5, visualize=visualize_octree)

        total_results["q_brdf_inv_sarsa_octree"] = renderer.render(**common_params, spatial_type='octree',
                                                                   sample_type=SAMPLE_Q_COS_PROPORTION,
                                                                   q_table_update_method=Q_UPDATE_SARSA)
        total_results["q_brdf_inv_sarsa_grid"] = renderer.render(**common_params, spatial_type='grid',
                                                                    sample_type=SAMPLE_Q_COS_PROPORTION,
                                                                    q_table_update_method=Q_UPDATE_SARSA)

    if test_target == 1:
        test_1()
    elif test_target == 2:
        test_2()
    elif test_target == 3:
        test_3()
    elif test_target == 4:
        test_4()
    elif test_target == 5:
        test_5()
    elif test_target == 6:
        test_6()

    if show_result:
        show_result_bar(total_results, "error_mean")
        show_result_bar(total_results, "elapsed_time_per_sample")
        show_result_bar(total_results, "elapsed_time_per_sample_except_init")

        show_result_bar(total_results, "total_hit_percentage")
        if test_time:
            show_result_bar(total_results, "completed_samples")
        show_result_sequence(total_results, "hit_count_sequence")
        show_result_sequence(total_results, "elapsed_times")
        show_result_bar(total_results, "invalid_sample_rate")

    if output_folder is not None:
        scene_output_folder = "%s/%s" % (output_folder, scene_name)

        # export images
        if not os.path.exists(scene_output_folder):
            os.makedirs(scene_output_folder)
        for k, v in total_results.items():
            save_pred_images(v['image'], "%s/images/%s" % (scene_output_folder, k))

        # export csv
        df = pd.DataFrame(total_results)

        def export_list_type(target):
            target_sequence = df.loc[target]
            df_target_sequence = pd.DataFrame({ key:pd.Series(value) for key, value in target_sequence.items()})
            df_target_sequence = df_target_sequence.transpose()
            df_target_sequence.to_csv("%s/%s.csv" % (scene_output_folder, target))

        export_list_type("elapsed_times")
        export_list_type("hit_count_sequence")

        df.drop(["image", "elapsed_times", "hit_count_sequence", "q_table_info"], inplace=True)
        df.to_csv("%s/result.csv" % scene_output_folder)

        # export json
        with open('%s/setting.json' % scene_output_folder, 'w') as fp:
            json.dump(common_params, fp)
        return df

    return total_results


def test_multiple_and_export_result(scene_list, scale, output_folder, _time=5, _spp=256, test_time=False, test_target=1):
    for scene in scene_list:
        #try:
        test_single_scene(scene, scale, test_time=test_time, show_result=False, _time=_time, _spp=_spp, output_folder=output_folder, test_target=test_target)
        # except Exception:
        #     print("Scene Error")

    update_total_result(output_folder)


def test_octree_build(scene_name, scale=2, visualize=False):
    #test_single_scene(scene_name, test_target=6)
    renderer = Renderer(scale=scale)
    renderer.construct_stree(scene_name, max_octree_depth=6, visualize=visualize)
    # renderer.render(scene_name, spatial_type='octree',
    #                 show_picture=True,
    #                 uv_n=8,
    #                 sample_type=SAMPLE_Q_COS_PROPORTION,
    #                 q_table_update_method=Q_UPDATE_SARSA)

