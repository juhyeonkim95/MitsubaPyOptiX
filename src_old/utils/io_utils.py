import os
import pandas as pd


def export_single_value(result_grouped, output_folder, key):
    # export error
    result = result_grouped.get_group(key)
    result = result.reset_index(level=[1])
    result = result.drop("value", axis=1)
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
    export_single_value(result_grouped, output_folder, "total_hit_percentage")

    if test_time:
        export_single_value(result_grouped, output_folder, "completed_samples")
