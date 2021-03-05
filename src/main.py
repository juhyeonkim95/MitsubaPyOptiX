# from renderer import *
# import matplotlib.pyplot as plt
# import datetime
#
#
# def test():
# 	renderer = Renderer()
# 	#scene_name = "material-testball"
# 	scene_name = "bathroom"
# 	image = renderer.render(scene_name, spp=256, use_mis=False)
#
# 	plt.imshow(image)
# 	plt.show()
#
# 	#file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# 	#save_pred_images(image, "../data/reference_images/%s_%s" % (scene_name,file_name))
#
#
# if __name__ == '__main__':
# 	test()

from core.renderer import *
import datetime
from utils.image_utils import *
from utils.io_utils import *

from collections import OrderedDict
import pandas as pd
import json


def make_reference_image_multiple(scale=1, scene_names=None, diffuse_only=False):
	renderer = Renderer(scale=scale)
	root_path = "../scene"
	if scene_names is None:
		scene_names = [os.path.relpath(f.path, root_path) for f in os.scandir(root_path) if f.is_dir()]
		scene_names.sort()
	print(scene_names)
	diffuse_folder = "diffuse_only" if diffuse_only else "standard"
	target_folder = '../reference_images/%s/scale_%d' % (diffuse_folder, scale)

	if not os.path.exists(target_folder):
		os.makedirs(target_folder)
	for scene_name in scene_names:
		common_params = {
			'scene_name': scene_name,
			'spp': 4096 * 16,
			'samples_per_pass': 128 * scale * scale,
			'max_depth': 16,
			'rr_begin_depth': 16,
		}
		try:
			image = renderer.render(**common_params, use_mis=False)
			file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
			save_pred_images(image['image'], "%s/%s_%s" % (target_folder, scene_name, file_name))
		except Exception:
			print("Error")


def make_reference_image_single(scene_name, scale=4, force_all_diffuse=False):
	renderer = Renderer(scale=scale, force_all_diffuse=force_all_diffuse)
	common_params = {
		'scene_name': scene_name,
		'spp': 1024,
		'samples_per_pass': 32,
		'max_depth': 8,
		'rr_begin_depth': 8,

		# You should change q_table_old at getQValue to q_table
		'accumulative_q_table_update': True
	}
	image = renderer.render(**common_params, use_mis=False, show_picture=True)
	file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

	#save_pred_images(image['image'], "../reference_images2/%s_%s" % (scene_name, file_name))


def test_multiple_and_export_result(scene_list, scale, output_folder, test_time=False):
	for scene in scene_list:
		try:
			test(scene, scale, test_time=test_time, show_result=False, output_folder=output_folder)
		except Exception:
			print("Scene Error")

	update_total_result(output_folder)


def test(scene_name, scale=4, test_time=False, show_picture=False, show_result=False,
		 output_folder=None, force_all_diffuse=False):
	diffuse_folder = "diffuse_only" if force_all_diffuse else "standard"
	reference_parent_folder = '../reference_images/%s/scale_%d' % (diffuse_folder, scale)

	ref_image = load_reference_image(reference_parent_folder, scene_name)

	total_results = OrderedDict()
	renderer = Renderer(scale=scale, force_all_diffuse=force_all_diffuse)
	renderer.reference_image = ref_image

	common_params = {
		'scene_name': scene_name,
		'samples_per_pass': 16,
		'show_picture': show_picture,
		'max_depth': 16,
		'rr_begin_depth': 8,
		'scene_epsilon': 1e-5,
		# You should change q_table_old at getQValue to q_table
		'accumulative_q_table_update': True
	}

	if test_time:
		time_limit_in_secs = {1:60, 2:20, 4:5}
		common_params['time_limit_in_sec'] = time_limit_in_secs[scale]
	else:
		common_params['spp'] = 1024
	common_params['time_limit_init_ignore_step'] = 10

	total_results["brdf"] = renderer.render(**common_params, sample_type=SAMPLE_COSINE)
	total_results["q"] = renderer.render(**common_params, sample_type=SAMPLE_Q_PROPORTION)
	total_results["q_brdf"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_PROPORTION)
	total_results["q_brdf_sarsa"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_PROPORTION, q_table_update_method=Q_UPDATE_SARSA)
	total_results["q_brdf_rej"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_REJECT)
	total_results["q_brdf_rej_sarsa"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_REJECT, q_table_update_method=Q_UPDATE_SARSA)
	total_results["q_brdf_mcmc"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_MCMC)
	total_results["q_brdf_mcmc_sarsa"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_MCMC, q_table_update_method=Q_UPDATE_SARSA)

	if show_result:
		show_result_bar(total_results, "error_mean")
		show_result_bar(total_results, "elapsed_time_per_spp")
		show_result_bar(total_results, "elapsed_time_except_init")

		show_result_bar(total_results, "total_hit_percentage")
		if test_time:
			show_result_bar(total_results, "completed_samples")
		show_result_sequence(total_results, "hit_count_sequence")
		show_result_sequence(total_results, "elapsed_times")

	if output_folder is not None:
		scene_output_folder = "%s/%s" % (output_folder, scene_name)

		# export images
		if not os.path.exists(scene_output_folder):
			os.makedirs(scene_output_folder)
		for k, v in total_results.items():
			save_pred_images(v['image'], "%s/images/%s" % (scene_output_folder, k))

		# export csv
		df = pd.DataFrame(total_results)
		df.drop(["image", "elapsed_times", "hit_count_sequence"], inplace=True)
		df.to_csv("%s/result.csv" % scene_output_folder)

		# export json
		with open('%s/setting.json' % scene_output_folder, 'w') as fp:
			json.dump(common_params, fp)
		return df

	return total_results


def test2(scene_name, scale=4, test_time=False, show_picture=False, show_result=False,
		 output_folder=None, force_all_diffuse=False):
	diffuse_folder = "diffuse_only" if force_all_diffuse else "standard"
	reference_parent_folder = '../reference_images/%s/scale_%d' % (diffuse_folder, scale)

	ref_image = load_reference_image(reference_parent_folder, scene_name)

	total_results = OrderedDict()
	renderer = Renderer(scale=scale, force_all_diffuse=force_all_diffuse)
	renderer.reference_image = ref_image

	common_params = {
		'scene_name': scene_name,
		'samples_per_pass': 32,
		'show_picture': show_picture,
		'max_depth': 8,
		'rr_begin_depth': 8,
		'scene_epsilon': 1e-3,
		# You should change q_table_old at getQValue to q_table
		'accumulative_q_table_update': True
	}

	if test_time:
		time_limit_in_secs = {1:60, 2:40, 4:10}
		common_params['time_limit_in_sec'] = time_limit_in_secs[scale]
	else:
		common_params['spp'] = 1024
	common_params['time_limit_init_ignore_step'] = 10

	total_results["brdf"] = renderer.render(**common_params, sample_type=SAMPLE_COSINE)
	#total_results["q"] = renderer.render(**common_params, sample_type=SAMPLE_Q_PROPORTION)
	# total_results["q_brdf"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_PROPORTION)
	# total_results["q_brdf_sarsa"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_PROPORTION, q_table_update_method=Q_UPDATE_SARSA)
	# total_results["q_brdf_rej"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_REJECT)
	# total_results["q_brdf_rej_sarsa"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_REJECT, q_table_update_method=Q_UPDATE_SARSA)
	# total_results["q_brdf_mcmc"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_MCMC)
	# total_results["q_brdf_mcmc_sarsa"] = renderer.render(**common_params, sample_type=SAMPLE_Q_COS_MCMC, q_table_update_method=Q_UPDATE_SARSA)

	if show_result:
		show_result_bar(total_results, "error_mean")
		show_result_bar(total_results, "elapsed_time_per_spp")
		show_result_bar(total_results, "elapsed_time_except_init")

		show_result_bar(total_results, "total_hit_percentage")
		if test_time:
			show_result_bar(total_results, "completed_samples")
		show_result_sequence(total_results, "hit_count_sequence")
		show_result_sequence(total_results, "elapsed_times")

	if output_folder is not None:
		scene_output_folder = "%s/%s" % (output_folder, scene_name)

		# export images
		if not os.path.exists(scene_output_folder):
			os.makedirs(scene_output_folder)
		for k, v in total_results.items():
			save_pred_images(v['image'], "%s/images/%s" % (scene_output_folder, k))

		# export csv
		df = pd.DataFrame(total_results)
		df.drop(["image", "elapsed_times", "hit_count_sequence"], inplace=True)
		df.to_csv("%s/result.csv" % scene_output_folder)

		# export json
		with open('%s/setting.json' % scene_output_folder, 'w') as fp:
			json.dump(common_params, fp)
		return df

	return total_results


if __name__ == '__main__':
	Compiler.add_program_directory(dirname(__file__))
	make_reference_image_single("cornell-box", scale=2, force_all_diffuse=False)
	#test2("cornell-box", scale=2, force_all_diffuse=False, show_picture=True)
	#make_reference_image_single("veach-ajar", scale=4, force_all_diffuse=True)

	#make_reference_image_single("veach_door_simple", scale=2)
	#make_reference_image_multiple(scale=1, scene_names=["veach_door_simple"])
	all_scenes = ['bathroom', 'bathroom2',  'bedroom', 'cornell-box', 'kitchen',
				  'living-room', 'living-room-2', 'living-room-3', 'staircase','staircase2',
				  'veach-ajar',"veach_door_simple"]
	#test_multiple_and_export_result(all_scenes, 1, "../result_accum/scale_1_spp_1024", test_time=False)
	#test_multiple_and_export_result(all_scenes, 1, "../result_accum/scale_1_time_60", test_time=True)
	#test_multiple_and_export_result(all_scenes, 2, "../result_non_accum/scale_2_time_40", test_time=True)
	#test_multiple_and_export_result(all_scenes, 2, "../result_non_accum/scale_2_spp_1024", test_time=False)
	#test_multiple_and_export_result(all_scenes, 1, "../result_accum/scale_1_spp_1024", test_time=False)
	#test_multiple_and_export_result(all_scenes, 1, "../result_accum/scale_1_spp_1024", test_time=True)

	#test_multiple_and_export_result(all_scenes, 4, "../result_accum/scale_4_time_5", test_time=True)

	#update_total_result("../result/scale_4")
	#all_scenes2 = ['living-room-2', 'living-room-3', 'staircase2', 'veach-ajar']
	# all_scenes = ['cornell-box']
	# make_reference_image(scale=8, scene_names=all_scenes)
	#make_reference_image_multiple(scale=4, scene_names=all_scenes)
	#make_reference_image_multiple(scale=2, scene_names=all_scenes)
	# make_reference_image(scale=1, scene_names=all_scenes)

	# make_reference_image_single("bathroom")

	#test2("veach_door_simple", scale=4, show_result=True, show_picture=True, test_time=False, force_all_diffuse=False)
	#test2("bathroom", scale=4, show_result=True, show_picture=True, test_time=False, force_all_diffuse=False)

	#make_reference_image_single("cornell-box-hard", scale=2)
	#test("cornell-box", scale=2)
	#make_reference_image_single("staircase")
	#test("bedroom")
