import multiprocessing
from multiprocessing import Queue, Lock, Process
from core.renderer_constants import *

import json
from utils.image_utils import *
from utils.io_utils import *

from collections import OrderedDict

renderers = {}


def render_using_multiprocessing(config_file_name):
    m = MultiProcessingRenderer()
    m.load_config(config_file_name)
    m.render()


def load_config_recursive(config_file_name):
    config = json.load(open(config_file_name))
    if "include" in config:
        config_include = load_config_recursive(config["include"])
        config.pop("include")
        merged = {**config, **config_include}
        return merged
    else:
        return config


class MultiProcessingRenderer:
    def __init__(self):
        self.rendered_result_dict = {}
        self.scene_list = []
        self.config_list = []
        self.config_name_list = []
        self.available_gpu_list = []
        self.common_configs = None
        self.json_config = None

    def set_config_name(self):
        # set each config name
        self.config_name_list = []
        for i in range(len(self.config_list)):
            config_i = self.config_list[i]
            config_name = config_i.get("name")
            if config_name is None:
                config_name = config_i["sample_type"]
                if "q_table_update_method" in config_i:
                    config_name += ("_" + config_i["q_table_update_method"])
                if config_i["sample_type"] == "q_mis_quadtree":
                    config_name += ("_" + config_i["quad_tree_update_type"])
            self.config_name_list.append(config_name)
        print("Config names", self.config_name_list)

    def load_config(self, config_file_name):
        config = load_config_recursive(config_file_name)

        self.json_config = config
        self.scene_list = config["scene"]

        # read from another file
        if type(self.config_list) == str:
            self.config_list = json.load(open(self.config_list))["config_list"]
        else:
            self.config_list = config["config_list"]

        self.available_gpu_list = config["available_gpu_ids"]

        self.set_config_name()

        for i in range(len(self.config_list)):
            process_configs(self.config_list[i])
        self.common_configs = config["common_config"]

    def render(self):
        # prepare queue and result store dictionary
        manager = multiprocessing.Manager()
        self.rendered_result_dict = manager.dict()
        for scene_name in self.scene_list:
            self.rendered_result_dict[scene_name] = manager.dict()

        queue = Queue()
        for scene_id in range(len(self.scene_list)):
            for config_id in range(len(self.config_list)):
                queue.put((scene_id, config_id))
        print("------Queue------", queue)
        lock = Lock()

        procs = []
        n_process = len(self.available_gpu_list)

        # create export folder
        if "output_folder" in self.json_config:
            output_folder = self.json_config["output_folder"]
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

        # (1) More than one GPUs --> multiprocessing
        if n_process > 1:
            for i in range(n_process):
                gpu_id = self.available_gpu_list[i]
                proc = Process(target=self.render_single_process, args=(gpu_id, queue, lock))
                procs.append(proc)
                proc.start()

            for proc in procs:
                proc.join()

        # (2) only one GPU --> no multiprocessing
        else:
            self.render_single_process(self.available_gpu_list[0], queue, lock)

        # export final result over scenes
        if "output_folder" in self.json_config:
            output_folder = self.json_config["output_folder"]
            update_total_result(output_folder)

    def render_single_process(self, gpu_id, queue, lock):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        from pyoptix import Compiler
        Compiler.clean()
        Compiler.keep_device_function = False
        file_dir = os.path.dirname(os.path.abspath(__file__))
        Compiler.add_program_directory(file_dir)

        from core.renderer import Renderer

        scale = self.common_configs.get("scale")
        renderer = Renderer(scale=scale)

        while True:
            with lock:
                if queue.empty():
                    break
                else:
                    scene_id, config_id = queue.get()
                    print("------------Popped------------", scene_id, config_id)

            scene_name = self.scene_list[scene_id]
            config = self.config_list[config_id]

            config_name = self.config_name_list[config_id]

            final_config = {**self.common_configs, **config}
            final_config.pop("scale")
            final_config["scene_name"] = scene_name

            reference_parent_folder = '../reference_images/%s/scale_%d' % ("standard", scale)
            ref_image = load_reference_image(reference_parent_folder, scene_name)
            renderer.reference_image = ref_image

            render_result = renderer.render(**final_config)

            with lock:
                self.add_rendered_result(scene_name, config_name, render_result)

    def add_rendered_result(self, scene_name, config_name, result):
        self.rendered_result_dict[scene_name][config_name] = result

        # export result if full
        if len(self.rendered_result_dict[scene_name]) == len(self.config_list):
            total_results = OrderedDict()
            for c_name in self.config_name_list:
                total_results[c_name] = self.rendered_result_dict[scene_name][c_name]

            if "output_folder" in self.json_config:
                output_folder = self.json_config["output_folder"]
                scene_output_folder = "%s/%s" % (output_folder, scene_name)
                export_total_results(total_results, scene_output_folder)

            del self.rendered_result_dict[scene_name]


def process_configs(config):
    sample_type_dict = {
        "brdf": SAMPLE_COSINE,
        "q_brdf_inv": SAMPLE_Q_COS_PROPORTION,
        "q_brdf_rej": SAMPLE_Q_COS_REJECT,
        "q_brdf_rej_mix": SAMPLE_Q_COS_REJECT_MIX,
        "q_brdf_mcmc": SAMPLE_Q_COS_MCMC,
        "q_mis_sphere": SAMPLE_Q_SPHERE,
        "q_mis_quadtree": SAMPLE_Q_QUADTREE
    }
    q_table_update_method_dict = {
        "sarsa": Q_UPDATE_SARSA,
        "expected_sarsa": Q_UPDATE_EXPECTED_SARSA,
        "mc": Q_UPDATE_MONTE_CARLO
    }
    if config.get("sample_type") is not None:
        config["sample_type"] = sample_type_dict[config["sample_type"]]
    if config.get("q_table_update_method") is not None:
        config["q_table_update_method"] = q_table_update_method_dict[config["q_table_update_method"]]
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

    df.drop(["image", "elapsed_times", "hit_count_sequence"], inplace=True)
    if "q_table_info" in df:
        df.drop(["q_table_info"], inplace=True)
    df.to_csv("%s/result.csv" % scene_output_folder)

    # export json
    # with open('%s/setting.json' % scene_output_folder, 'w') as fp:
    #    json.dump(common_params, fp)
    return df
