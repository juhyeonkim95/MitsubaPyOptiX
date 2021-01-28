from main import *

def make_reference(scene_name, n):
    set_scene(scene_name, n)
    render(SAMPLE_COSINE, show_picture=True, save_image=True,
           save_image_name="%s_%d" % (scene_name, n*n), use_mis=True, samples_per_pass=64)

def make_reference_foggy(scene_name, sigma_s, hg_g, n):
    set_scene(scene_name, n, _sigma_s=sigma_s, _hg_g=hg_g)
    render(SAMPLE_COSINE, show_picture=True, scatter_sample_type=SAMPLE_HG, save_image=True,
           save_image_name="%s_foggy_sigma_s_%.4f_hg_%.4f_%d" % (scene_name, sigma_s, hg_g, n*n), use_mis=False)


def test_scene(scene_name, n, reference_image_n=0, export_result=False, save_image=False):
    set_scene(scene_name, n)
    reference_image_name = None
    if reference_image_n != 0:
        reference_image_name = "%s_%d" % (scene_name, reference_image_n * reference_image_n)
        load_reference_image(reference_image_name)
    reference_image_name = scene_name if reference_image_name is None else reference_image_name

    show_picture = True
    total_results = OrderedDict()
    # total_results["brdf"] = render(SAMPLE_UNIFORM, show_picture=show_picture,
    #                                save_image=False, use_mis=False,
    #                                samples_per_pass=8, export_video=True, export_video_name="brdf")

    # total_results["q_brdf"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture,
    #                                  save_image=False, use_mis=False, samples_per_pass=1,
    #                                  accumulative_q_table_update=True)
    s = 8
    # total_results["uniform"] = render(SAMPLE_UNIFORM, show_picture=show_picture, save_image=False, use_mis=False, samples_per_pass=1)
    total_results["brdf"] = render(SAMPLE_COSINE, show_picture=show_picture, save_image=False, use_mis=False, samples_per_pass=s)
    total_results["q"] = render(SAMPLE_Q_PROPORTION, show_picture=show_picture, save_image=False,
                                use_mis=False, samples_per_pass=s, accumulative_q_table_update=True,
                                save_q_cos=False)
    # total_results["q2"] = render(SAMPLE_Q_PROPORTION, show_picture=show_picture, save_image=False,
    #                             use_mis=False, samples_per_pass=1, accumulative_q_table_update=True,
    #                             save_q_cos=True)

    total_results["q_brdf"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture, save_image=False,
                                     use_mis=False, samples_per_pass=s, accumulative_q_table_update=True)

    total_results["q_cos_mis"] = render(SAMPLE_Q_COS_MIS, show_picture=show_picture, save_image=False,
                                        use_mis=False, samples_per_pass=s, accumulative_q_table_update=True)
    total_results["q_cos_mis2"] = render(SAMPLE_Q_COS_MIS, show_picture=show_picture, save_image=False,
                                        q_table_update_method=Q_UPDATE_SARSA,
                                        use_mis=False, samples_per_pass=s, accumulative_q_table_update=True)

    # total_results["q_cos_mis2"] = render(SAMPLE_Q_COS_MIS, show_picture=show_picture, save_image=False,
    #                                     use_mis=False, samples_per_pass=1, accumulative_q_table_update=True)

    #total_results["q_brdf"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture, save_image=False,
    #                                       use_mis=False, samples_per_pass=1, accumulative_q_table_update=False)

    #total_results["q"] = render(SAMPLE_Q_PROPORTION, show_picture=show_picture, save_image=False, use_mis=False, samples_per_pass=1)
    #total_results["q_brdf"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture, save_image=False, use_mis=False, samples_per_pass=1)

    # total_results["q_brdf_2"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture, save_image=False, use_mis=False, samples_per_pass=2)
    # total_results["q_brdf_4"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture, save_image=False, use_mis=False, samples_per_pass=4)
    # total_results["q_brdf_8"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture, save_image=False, use_mis=False, samples_per_pass=8)
    # total_results["q_brdf_16"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture, save_image=False, use_mis=False, samples_per_pass=16)
    # total_results["q_brdf_32"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture, save_image=False, use_mis=False, samples_per_pass=32)
    # total_results["q_brdf_64"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture, save_image=False, use_mis=False, samples_per_pass=64)
    # total_results["q_brdf_128"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture, save_image=False, use_mis=False, samples_per_pass=128)
    # total_results["q_brdf_256"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture, save_image=False, use_mis=False, samples_per_pass=256)

    #total_results["q_brdf_exp_discard"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture, save_image=False, use_mis=False, learning_method='exponential', sample_combination='discard')
    #total_results["q_brdf_exp_inv"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture, save_image=False, use_mis=False, learning_method='exponential', sample_combination='inverse_variance')
    #total_results["q_brdf_exp_none"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture, save_image=False, use_mis=False, learning_method='exponential', sample_combination='none')

    # total_results["sarsa"] = render(SAMPLE_Q_COS_PROPORTION, q_table_update_method=Q_UPDATE_SARSA, show_picture=True, save_image=False, use_mis=False)
    # total_results["q_learning"] = render(SAMPLE_Q_COS_PROPORTION,q_table_update_method=Q_UPDATE_SARSA, show_picture=True, save_image=False, use_mis=False)
    # total_results["ex_sarsa"] = render(SAMPLE_Q_COS_PROPORTION, q_table_update_method=Q_UPDATE_SARSA, show_picture=True, save_image=False, use_mis=False)

    show_result_graph(total_results, "error_mean")
    show_result_graph(total_results, "total_hit_percentage")
    show_result_graph(total_results, "elapsed_time_per_spp")
    show_hit_percentage_sequence(total_results)

    if export_result:
        save_result(reference_image_name, total_results)
    if save_image:
        save_images(reference_image_name, total_results)


def test_scene_sarsa(scene_name, n, reference_image_n=0, export_result=False, save_image=False):
    set_scene(scene_name, n)
    reference_image_name = None
    if reference_image_n != 0:
        reference_image_name = "%s_%d" % (scene_name, reference_image_n * reference_image_n)
        load_reference_image(reference_image_name)
    reference_image_name = scene_name if reference_image_name is None else reference_image_name

    total_results = OrderedDict()
    total_results["sarsa"] = render(SAMPLE_Q_COS_PROPORTION, q_table_update_method=Q_UPDATE_SARSA, show_picture=True, save_image=False, use_mis=False)
    total_results["q_learning"] = render(SAMPLE_Q_COS_PROPORTION,q_table_update_method=Q_UPDATE_Q_LEARNING, show_picture=True, save_image=False, use_mis=False)
    total_results["ex_sarsa"] = render(SAMPLE_Q_COS_PROPORTION, q_table_update_method=Q_UPDATE_EXPECTED_SARSA, show_picture=True, save_image=False, use_mis=False)

    show_result_graph(total_results, "error_mean")
    show_result_graph(total_results, "total_hit_percentage")
    show_result_graph(total_results, "elapsed_time_per_spp")
    show_hit_percentage_sequence(total_results)

    if export_result:
        save_result(reference_image_name+"_sarsa", total_results)
    if save_image:
        save_images(reference_image_name+"_sarsa", total_results)

def test_scene_sarsa_foggy(scene_name, sigma_s, hg_g, n, reference_image_n=0, export_result=False, save_image=False):
    set_scene(scene_name, n, _sigma_s=sigma_s, _hg_g=hg_g)
    reference_image_name = None
    if reference_image_n != 0:
        reference_image_name = "%s_foggy_sigma_s_%.4f_hg_%.4f_%d" % (scene_name, sigma_s, hg_g, reference_image_n * reference_image_n)
        load_reference_image(reference_image_name)
    reference_image_name = scene_name if reference_image_name is None else reference_image_name

    total_results = OrderedDict()
    show_picture = True
    total_results["sarsa"] = render(SAMPLE_Q_COS_PROPORTION, q_table_update_method=Q_UPDATE_SARSA,show_picture=show_picture, scatter_sample_type=SAMPLE_Q_HG_PROPORTION)
    total_results["q_learning"] = render(SAMPLE_Q_COS_PROPORTION, q_table_update_method=Q_UPDATE_Q_LEARNING,show_picture=show_picture, scatter_sample_type=SAMPLE_Q_HG_PROPORTION)
    total_results["ex_sarsa"] = render(SAMPLE_Q_COS_PROPORTION, q_table_update_method=Q_UPDATE_EXPECTED_SARSA, show_picture=show_picture, scatter_sample_type=SAMPLE_Q_HG_PROPORTION)

    show_result_graph(total_results, "error_mean")
    show_result_graph(total_results, "total_hit_percentage")
    show_result_graph(total_results, "elapsed_time_per_spp")
    show_hit_percentage_sequence(total_results)

    if export_result:
        save_result(reference_image_name+"_sarsa", total_results)
    if save_image:
        save_images(reference_image_name+"_sarsa", total_results)

def test_all_volumetric_algorithms():
    total_results = OrderedDict()
    show_picture = False
    total_results["uniform_uniform"] = render(SAMPLE_UNIFORM, show_picture=show_picture, scatter_sample_type=SAMPLE_UNIFORM)
    total_results["uniform_phase"] = render(SAMPLE_UNIFORM, show_picture=show_picture, scatter_sample_type=SAMPLE_HG)
    total_results["uniform_q"] = render(SAMPLE_UNIFORM, show_picture=show_picture, scatter_sample_type=SAMPLE_Q_PROPORTION)
    total_results["uniform_q_phase"] = render(SAMPLE_UNIFORM, show_picture=show_picture, scatter_sample_type=SAMPLE_Q_HG_PROPORTION)

    total_results["brdf_uniform"] = render(SAMPLE_COSINE, show_picture=show_picture, scatter_sample_type=SAMPLE_UNIFORM)
    total_results["brdf_phase"] = render(SAMPLE_COSINE, show_picture=show_picture, scatter_sample_type=SAMPLE_HG)
    total_results["brdf_q"] = render(SAMPLE_COSINE, show_picture=show_picture, scatter_sample_type=SAMPLE_Q_PROPORTION)
    total_results["brdf_q_phase"] = render(SAMPLE_COSINE, show_picture=show_picture, scatter_sample_type=SAMPLE_Q_HG_PROPORTION)

    total_results["q_uniform"] = render(SAMPLE_Q_PROPORTION, show_picture=show_picture, scatter_sample_type=SAMPLE_UNIFORM)
    total_results["q_phase"] = render(SAMPLE_Q_PROPORTION, show_picture=show_picture, scatter_sample_type=SAMPLE_HG)
    total_results["q_q"] = render(SAMPLE_Q_PROPORTION, show_picture=show_picture, scatter_sample_type=SAMPLE_Q_PROPORTION)
    total_results["q_q_phase"] = render(SAMPLE_Q_PROPORTION, show_picture=show_picture, scatter_sample_type=SAMPLE_Q_HG_PROPORTION)

    total_results["q_brdf_uniform"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture, scatter_sample_type=SAMPLE_UNIFORM)
    total_results["q_brdf_phase"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture, scatter_sample_type=SAMPLE_HG)
    total_results["q_brdf_q"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture, scatter_sample_type=SAMPLE_Q_PROPORTION)
    total_results["q_brdf_q_phase"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture,  scatter_sample_type=SAMPLE_Q_HG_PROPORTION)
    return total_results


def test_all_volumetric_algorithms2():
    total_results = OrderedDict()
    show_picture = True
    total_results["uniform"] = render(SAMPLE_UNIFORM, show_picture=show_picture, scatter_sample_type=SAMPLE_UNIFORM)
    total_results["brdf_phase"] = render(SAMPLE_COSINE, show_picture=show_picture, scatter_sample_type=SAMPLE_HG)
    total_results["q"] = render(SAMPLE_Q_PROPORTION, show_picture=show_picture, scatter_sample_type=SAMPLE_Q_PROPORTION)
    total_results["q_brdf_phase"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture,  scatter_sample_type=SAMPLE_Q_HG_PROPORTION)
    return total_results

def test_scene_foggy(scene_name, sigma_s, hg_g, n, reference_image_n=0, export_result=False, save_image=False):
    set_scene(scene_name, n, _sigma_s=sigma_s, _hg_g=hg_g)
    reference_image_name = None
    if reference_image_n != 0:
        reference_image_name = "%s_foggy_sigma_s_%.4f_hg_%.4f_%d" % (scene_name, sigma_s, hg_g, reference_image_n * reference_image_n)
        load_reference_image(reference_image_name)
    reference_image_name = scene_name if reference_image_name is None else reference_image_name

    total_results = test_all_volumetric_algorithms()
    show_result_graph(total_results, "error_mean")
    show_result_graph(total_results, "total_hit_percentage")
    show_result_graph(total_results, "elapsed_time_per_spp")
    show_hit_percentage_sequence(total_results)

    if export_result:
        save_result(reference_image_name, total_results)
    if save_image:
        save_images(reference_image_name, total_results)


def make_ref_and_test_scene(scene_name, ref_image_n, test_n):
    make_reference(scene_name, ref_image_n)
    test_scene(scene_name, test_n, "%s_%d" % (scene_name, ref_image_n * ref_image_n))


def make_ref_and_test_scene_foggy(scene_name, ref_image_n, test_n, sigma_s, hg_g, make_reference=True):
    if make_reference:
        make_reference_foggy(scene_name, sigma_s, hg_g, ref_image_n)
    saved_scene_name = "%s_foggy_sigma_s_%.4f_hg_%.4f_%d" % (scene_name, sigma_s, hg_g, ref_image_n*ref_image_n)
    test_scene_foggy(scene_name, sigma_s, hg_g, test_n, saved_scene_name)


def save_images(scene_name, total_results):
    os.makedirs("../result/%s" % scene_name, exist_ok=True)
    for k, v in total_results.items():
        image = v["final_image"]
        save_pred_images(image, "../result/%s/%s" % (scene_name, k))

import pandas as pd


def save_result(scene_name, total_results):
    os.makedirs("../result/%s" % scene_name, exist_ok=True)
    df = pd.DataFrame.from_dict(total_results, orient='index')
    df = df[["error_mean", "total_hit_percentage", "elapsed_time_per_spp"]]
    df.to_csv("../result/%s/%s.csv" % (scene_name, "total_result"))


if __name__ == '__main__':
    # 0. for program development test scene.
    # set_scene("cornell_box_sphere_light", 16)
    # render(SAMPLE_COSINE, show_picture=True, save_image=False, use_mis=True)

    # 1. cornell box
    # make_reference("cornell_box", 128)
    # make_reference("cornell_box_hard2", 128)
    # make_reference("cornell_box_hard3", 128)
    # test_scene("cornell_box", 16, 128, export_result=False, save_image=False)
    # test_scene("cornell_box_hard2", 16, 128, export_result=False, save_image=False)
    # test_scene("cornell_box", 16, 128, export_result=False, save_image=False)

    # 2. cornell box foggy
    # make_reference_foggy("cornell_box", 0.2, 0.7, 128)
    # make_reference_foggy("cornell_box_hard2", 0.2, 0.7, 128)
    # test_scene_foggy("cornell_box", 0.2, 0.7, 16, 128, export_result=True, save_image=True)
    # test_scene_foggy("cornell_box", 0.2, 0.7, 16, 128, export_result=False, save_image=False)

    # 3. veach door simple
    # make_reference("veach-ajar", 16)
    # make_reference("veach_door_simple", 128)
    test_scene("veach_door_simple", 32, 128, export_result=False, save_image=False)
    # make_reference_foggy("veach_door_simple", 0.001, 0.7, 128)
    # make_reference_foggy("veach_door_simple", 0.003, 0.7, 128)
    # test_scene_foggy("veach_door_simple", 0.001, 0.7, 32, 128, export_result=False, save_image=False)
    # test_scene_foggy("veach_door_simple", 0.003, 0.7, 16, 128, export_result=True, save_image=True)

    # 4. SARSA
    #test_scene_sarsa("cornell_box", 32, 128, export_result=True, save_image=True)
    #test_scene_sarsa("veach_door_simple", 32, 128, export_result=True, save_image=True)
    #test_scene_sarsa_foggy("veach_door_simple", 0.2, 0.7, 32, 128, export_result=True, save_image=True)
    #test_scene_sarsa_foggy("cornell_box_hard2", 0.2, 0.7, 32, 128, export_result=True, save_image=True)
    #test_scene_sarsa("veach_door_simple", 32, 128, export_result=True, save_image=True)
    #test_scene_sarsa_foggy("veach_door_simple", 0.2, 0.7, 32, 128, export_result=True, save_image=True)

    # 1. cornell box
    # make_ref_and_test_scene("cornell_box", 128, 16)
    # make_ref_and_test_scene("cornell_box_hard", 128, 16)
    # make_ref_and_test_scene("veach_door_simple", 128, 16)

    # print(111)
    # 2. cornell box foggy
    # make_ref_and_test_scene_foggy("cornell_box", 128, 16, 0.2, 0.7)
    # make_ref_and_test_scene_foggy("cornell_box_hard", 128, 16, 0.2, 0.7)
    # make_reference("veach_door_simple", 64)
    # make_reference_foggy("veach_door_simple", 0.02, 0.7, 16)

    #make_ref_and_test_scene_foggy("veach_door_simple", 128, 16, 0.003, 0.7, False)

    #make_reference("cornell_box", 16)
    #test_scene_with_no_medium("cornell_box", 16, "cornell_box_")
    #make_ref_and_test(0.2, 0.5)
    #make_ref_and_test(0.2, 0.6)
    #make_ref_and_test(0.2, 0.7)
    #make_ref_and_test(0.2, 0.8)
    #make_ref_and_test(0.2, 0.9)
    # make_reference("veach_door_simple", 16)

    #test_cornell_box_foggy(0.2, 0.9)
    #test_scene_foggy("veach_door_simple", 0.001, 0.6, 32, load_reference=False)
