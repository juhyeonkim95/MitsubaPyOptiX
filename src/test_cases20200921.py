from main import *


def make_reference(scene_name, n):
    set_scene(scene_name, n)
    render(SAMPLE_COSINE, show_picture=True, save_image=True,
           save_image_name="%s_%d" % (scene_name, n*n), use_mis=False)


def test_scene(scene_name, n, reference_image_n=0, export_result=False, save_image=False):
    set_scene(scene_name, n)
    reference_image_name = None
    if reference_image_n != 0:
        reference_image_name = "%s_%d" % (scene_name, reference_image_n * reference_image_n)
        load_reference_image(reference_image_name)
    reference_image_name = scene_name if reference_image_name is None else reference_image_name

    total_results = OrderedDict()
    show_picture = True
    total_results["uniform"] = render(SAMPLE_UNIFORM, show_picture=show_picture, save_image=False, use_mis=False)
    total_results["brdf"] = render(SAMPLE_COSINE, show_picture=show_picture, save_image=False, use_mis=False)
    total_results["q"] = render(SAMPLE_Q_PROPORTION, show_picture=show_picture, save_image=False, use_mis=False)
    total_results["q_brdf"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture, save_image=False, use_mis=False)
    total_results["q_soft"] = render(SAMPLE_Q_PROPORTION, show_picture=show_picture, save_image=False, use_mis=False, use_soft_q_update=True)
    total_results["q_brdf_soft"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture, save_image=False, use_mis=False, use_soft_q_update=True)

    show_result_graph(total_results, "error_mean")
    show_result_graph(total_results, "total_hit_percentage")
    show_result_graph(total_results, "elapsed_time_per_spp")
    show_hit_percentage_sequence(total_results)

    if export_result:
        save_result(reference_image_name, total_results)
    if save_image:
        save_images(reference_image_name, total_results)


def test_all_volumetric_algorithms():
    total_results = OrderedDict()
    show_picture = False
    total_results["uniform"] = render(SAMPLE_UNIFORM, show_picture=show_picture, scatter_sample_type=SAMPLE_UNIFORM)
    total_results["brdf_phase"] = render(SAMPLE_COSINE, show_picture=show_picture, scatter_sample_type=SAMPLE_HG)
    total_results["q"] = render(SAMPLE_Q_PROPORTION, show_picture=show_picture, scatter_sample_type=SAMPLE_Q_PROPORTION)
    total_results["q_brdf_phase"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=show_picture,  scatter_sample_type=SAMPLE_Q_HG_PROPORTION)
    total_results["q_soft"] = render(SAMPLE_Q_PROPORTION, show_picture=False, scatter_sample_type=SAMPLE_Q_PROPORTION, use_soft_q_update=True)
    total_results["q_brdf_soft"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=False, scatter_sample_type=SAMPLE_Q_HG_PROPORTION, use_soft_q_update=True)

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
    set_scene("cornell_box", 32)
    # render(SAMPLE_COSINE, show_picture=True, save_image=False, use_mis=False)
    # 1. cornell box
    make_reference("cornell_box", 64)
    #test_scene("cornell_box", 20, 128, export_result=False, save_image=False)
    #test_scene_foggy("cornell_box", 0.2, 0.7, 16, 128, export_result=True, save_image=True)
