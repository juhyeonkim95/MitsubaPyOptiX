# def render_single_using_multiprocessing(x):
#     kwargs, scale, gpu_id = x
#     renderer = Renderer(scale=scale)
#
#     render_result = renderer.render(**kwargs)
#     return render_result


import multiprocessing
from core.renderer_constants import *

import json
from utils.image_utils import *
from utils.result_export_utils import *

from collections import OrderedDict

renderers = {}


# def create_renderers(gpu_ids, scale):
#     from core.renderer import Renderer
#     for i in gpu_ids:
#         print("Context Created")
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
#         from pyoptix import Compiler
#         Compiler.clean()
#         Compiler.keep_device_function = False
#         file_dir = os.path.dirname(os.path.abspath(__file__))
#         Compiler.add_program_directory(file_dir)
#         renderer = Renderer(scale=scale)
#         renderers[i] = renderer

def render_single_using_multiprocessing(x):
    common_configs, gpu_id, config_list = x
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if gpu_id not in renderers:

        from pyoptix import Compiler
        Compiler.clean()
        Compiler.keep_device_function = False
        file_dir = os.path.dirname(os.path.abspath(__file__))
        Compiler.add_program_directory(file_dir)

        from core.renderer import Renderer

        scale = common_configs.get("scale")
        renderer = Renderer(scale=scale)
        renderers[gpu_id] = renderer

    renderer = renderers[gpu_id]
    config = {**common_configs, **config_list}
    config.pop("scale")
    render_result = renderer.render(**config)
    return render_result


def process_configs(config):
    sample_type_dict = {
        "brdf": SAMPLE_BRDF,
        "q_brdf_inv": SAMPLE_Q_COS_INVERSION,
        "q_brdf_rej": SAMPLE_Q_COS_REJECT,
        "q_brdf_rej_mix": SAMPLE_Q_COS_REJECT_MIX,
        #"q_brdf_mcmc": SAMPLE_Q_COS_MCMC,
        #"q_mis_sphere": SAMPLE_Q_SPHERE,
        #"q_mis_quadtree": SAMPLE_Q_QUADTREE
    }

    q_table_update_method_dict = {
        "sarsa": Q_UPDATE_SARSA,
        "expected_sarsa": Q_UPDATE_EXPECTED_SARSA,
        "mc": Q_UPDATE_MONTE_CARLO
    }
    if config.get("sampling_strategy") is not None:
        config["sampling_strategy"] = sample_type_dict[config["sampling_strategy"]]
    if config.get("q_table_update_method") is not None:
        config["q_table_update_method"] = q_table_update_method_dict[config["q_table_update_method"]]
    #if config.get("sampling_strategy") == "q_mis_quadtree":
    #
    return config


def export_total_results(total_results, scene_output_folder):
    # export images
    if not os.path.exists(scene_output_folder):
        os.makedirs(scene_output_folder)
    for k, v in total_results.items():
        save_pred_images(v['image'], "%s/images/%s" % (scene_output_folder, k))

    # export csv
    df = pd.DataFrame(total_results)

    def export_list_type(target):
        target_sequence = df.loc[target]
        df_target_sequence = pd.DataFrame({key: pd.Series(value) for key, value in target_sequence.items()})
        df_target_sequence = df_target_sequence.transpose()
        df_target_sequence.to_csv("%s/%s.csv" % (scene_output_folder, target))

    export_list_type("elapsed_times")
    export_list_type("hit_count_sequence")

    df.drop(["image", "elapsed_times", "hit_count_sequence", "q_table_info"], inplace=True)
    df.to_csv("%s/result.csv" % scene_output_folder)

    # export json
    # with open('%s/setting.json' % scene_output_folder, 'w') as fp:
    #    json.dump(common_params, fp)
    return df


def test_all_using_multiprocessing(config_file_name):

    # load config from json
    config = json.load(open(config_file_name))
    scene_list = config["scene"]

    available_gpu_ids = config["available_gpu_ids"]
    available_gpu_counts = len(available_gpu_ids)
    common_configs = config["common_config"]
    scale = common_configs.get("scale")

    # create_renderers(available_gpu_ids, scale)

    config_list = config["config_list"]
    config_counts = len(config_list)

    # set each config name
    config_name_list = []
    for i in range(config_counts):
        config_i = config_list[i]
        config_name = config_i.get("name")
        if config_name is None:
            config_name = config_i["sampling_strategy"]
            if "q_table_update_method" in config_i:
                config_name += ("_" + config_i["q_table_update_method"])
            if config_i["sampling_strategy"] == "q_mis_quadtree":
                config_name += ("_" + config_i["quad_tree_update_type"])
        config_name_list.append(config_name)
    print("Config names", config_name_list)

    gpu_ids = [available_gpu_ids[i % len(available_gpu_ids)] for i in range(config_counts)]


    for i in range(config_counts):
        process_configs(config_list[i])
    for scene in scene_list:
        render_infos = []
        common_configs["scene_name"] = scene
        for i in range(config_counts):
            #processed_config = process_configs(config_list[i])
            render_info = common_configs, gpu_ids[i], config_list[i]
            render_infos.append(render_info)


        with multiprocessing.Pool(available_gpu_counts) as p:
            total_results_list = p.map(render_single_using_multiprocessing, render_infos)

        total_results = OrderedDict()
        for i in range(config_counts):
            total_results[config_name_list[i]] = total_results_list[i]

        if "output_folder" in config:
            output_folder = config["output_folder"]
            scene_output_folder = "%s/%s" % (output_folder, scene)
            export_total_results(total_results, scene_output_folder)