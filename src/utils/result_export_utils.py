import os
import pandas as pd
from utils.image_utils import *


def export_single_value(result_grouped, output_folder, key):
    # export error
    result = result_grouped.get_group(key)
    result = result.reset_index(level=[1])
    result = result.drop("value", axis=1)
    for c in result.columns:
        result[c] = pd.to_numeric(result[c])
    result.loc['mean'] = result.mean()
    result.to_csv("%s/%s.csv" % (output_folder, key))


def update_total_result(output_folder, test_time=False):
    frames = []
    names = []

    list_subfolders_with_paths = [f.path for f in os.scandir(output_folder) if f.is_dir()]
    list_subfolders_with_paths.sort()
    for subfolder_name in list_subfolders_with_paths:
        scene_name = subfolder_name.split("/")[-1]
        df = pd.read_csv("%s/result.csv" % subfolder_name, index_col=0)
        frames.append(df)
        names.append(scene_name)

    total_frame = pd.concat(frames, keys=names, names=["scene name", "value"])
    result_grouped = total_frame.groupby("value")

    # export values
    export_single_value(result_grouped, output_folder, "error_mean")
    export_single_value(result_grouped, output_folder, "total_elapsed_time")
    export_single_value(result_grouped, output_folder, "elapsed_time_per_sample_except_init")
    export_single_value(result_grouped, output_folder, "elapsed_time_per_sample")
    export_single_value(result_grouped, output_folder, "total_hit_percentage")
    export_single_value(result_grouped, output_folder, "invalid_sample_rate")

    if test_time:
        export_single_value(result_grouped, output_folder, "completed_samples")


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

    export_list_type("hit_rate_per_pass")
    export_list_type("elapsed_time_per_sample_per_pass")
    export_list_type("q_table_update_times")

    df.drop(["image", "hit_rate_per_pass",
             "elapsed_time_per_sample_per_pass", "q_table_update_times"], inplace=True)
    if "q_table_info" in df:
        df.drop(["q_table_info"], inplace=True)
    df.to_csv("%s/result.csv" % scene_output_folder)

    # export json
    # with open('%s/setting.json' % scene_output_folder, 'w') as fp:
    # 	json.dump(common_params, fp)
    return df
