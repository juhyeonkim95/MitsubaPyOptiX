from os.path import dirname
#from pyoptix.matrix4x4 import Matrix4x4
import time
from datetime import timedelta
import numpy as np
from pyoptix import Context, Compiler, Buffer, Program, Geometry, Material, \
	GeometryInstance, EntryPoint, GeometryGroup, Acceleration, TextureSampler
import matplotlib.pyplot as plt
from PIL import Image
from core.scene import *
import datetime
from utils.math_utils import *
from core.light import Light
import cupy as cp
import torch
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import cv2

ESCAPE_KEY = 27

width = 512
height = 512
Compiler.add_program_directory(dirname(__file__))
target_program = 'optix/integrators/path_trace_camera.cu'
N_SQRT_NUM_SAMPLES = 16
N_SPP = N_SQRT_NUM_SAMPLES * N_SQRT_NUM_SAMPLES
N_CUBE = 16
UV_N = 8

SAMPLE_UNIFORM = 0
SAMPLE_COSINE = 1
SAMPLE_HG = 1
SAMPLE_Q_PROPORTION = 2
SAMPLE_Q_COS_PROPORTION = 3
SAMPLE_Q_COS_MIS= 4
SAMPLE_Q_HG_PROPORTION = 3


Q_UPDATE_EXPECTED_SARSA = 0
Q_UPDATE_Q_LEARNING = 1
Q_UPDATE_SARSA = 2

Q_SAMPLE_PROPORTIONAL_TO_Q = 1
Q_SAMPLE_PROPORTIONAL_TO_Q_SQUARE = 2

sigma_s = 0.0
sigma_a = 0.0
sigma_t = 0.0
hg_g = 0.9


use_upside_down_light = False
reference_image = None
reference_image_hdr = None

scene_name = "cornell_box"


def convert_image_to_uint(image):
	x = np.copy(image)
	x *= 255
	x = np.clip(x, 0, 255)
	x = x.astype('uint8')
	return x


def save_pred_images(images, file_path):
	x = convert_image_to_uint(images)
	new_im = Image.fromarray(x)
	new_im.save("%s.png" % file_path)


def set_camera(context, camera: Camera):
	fov = camera.fov
	aspect_ratio = float(width) / float(height)
	fovx = camera.fov_axis == 'x'

	# claculate camera variables
	W = np.array(camera.w)
	U = np.array(camera.u)
	V = np.array(camera.v)
	wlen = np.sqrt(np.sum(W ** 2))
	if fovx:
		ulen = wlen * math.tan(0.5 * fov * math.pi / 180)
		U *= ulen
		vlen = ulen / aspect_ratio
		V *= vlen
	else:
		vlen = wlen * math.tan(0.5 * fov * math.pi / 180)
		V *= vlen
		ulen = vlen * aspect_ratio
		U *= ulen

	context["eye"] = camera.eye
	context["U"] = U
	context["V"] = V
	context["W"] = W
	context["focalDistance"] = np.array(5, dtype=np.float32)
	#context["apertureRadius"] = np.array(0.5, dtype=np.float32)
	context["camera_type"] = np.array(0, dtype=np.uint32)


def render(
		sample_type=0,
		q_table_update_method=Q_UPDATE_EXPECTED_SARSA,
		q_value_sample_method=1,
		q_value_sample_constant=1.0,
		show_q_value_map=False,
		show_picture=True,
		save_image=False,
		save_image_name=None,
		scatter_sample_type=1,
		use_mis=False,
		use_tone_mapping=True,
		use_soft_q_update=False,
		samples_per_pass=16,
		learning_method="incremental",
		sample_combination="none",
		accumulative_q_table_update=False,
		export_video=False,
		export_video_name= None,
		save_q_cos=False
	):
	global width, height

	start_time = time.time()
	context, entry_point = create_context()
	context["sample_type"] = np.array(sample_type, dtype=np.uint32)
	context["use_mis"] = np.array(1 if use_mis else 0, dtype=np.uint32)
	context["use_tone_mapping"] = np.array(1 if use_tone_mapping else 0, dtype=np.uint32)
	context["use_soft_q_update"] = np.array(1 if use_soft_q_update else 0, dtype=np.uint32)
	context["save_q_cos"] = np.array(1 if save_q_cos else 0, dtype=np.uint32)

	context["scatter_sample_type"] = np.array(scatter_sample_type, dtype=np.uint32)
	context["q_table_update_method"] = np.array(q_table_update_method, dtype=np.uint32)
	context["q_value_sample_method"] = np.array(q_value_sample_method, dtype=np.uint32)
	context["q_value_sample_constant"] = np.array(q_value_sample_constant, dtype=np.float32)
	context["samples_per_pass"] = np.array(1, dtype=np.uint32)
	context["accumulative_q_table_update"] = np.array(1 if accumulative_q_table_update else 0, dtype=np.uint32)

	need_q_table_update = sample_type == SAMPLE_Q_PROPORTION \
				or sample_type == SAMPLE_Q_COS_PROPORTION \
				or scatter_sample_type == SAMPLE_Q_PROPORTION\
				or scatter_sample_type == SAMPLE_Q_HG_PROPORTION\
				or sample_type == SAMPLE_Q_COS_MIS
	context["need_q_table_update"] = np.array(1 if need_q_table_update else 0, dtype=np.uint32)
	# register_bsdf_programs(context)

	# context.set_print_enabled(True)
	# context.set_print_buffer_size(20000)
	scene = Scene(scene_name)
	scene.load_scene_from("../data/%s/scene.xml" % scene_name)
	scene.load_images()
	scene.create_objs()
	scene.create_object_instances()

	width = scene.width
	height = scene.height

	room_size = scene.bbox.bbox_max - scene.bbox.bbox_min
	print("Room Size:", room_size)
	q_table_original = create_q_table_related(context, room_size)

	create_geometry(context, scene)
	create_scene_lights(context, scene)
	set_sigmas(context)

	[n_a, n_s] = q_table_original.shape
	q_table = np.zeros((n_a, n_s), dtype=np.float32)
	entry_point.launch((width, height))

	equal_table = np.zeros((n_a, n_s), dtype=np.float32)
	equal_table.fill(1 / n_a)
	hit_sum = 0
	hit_counts = []

	# initial samples
	if learning_method == "exponential":
		current_samples_per_pass = 1
	else:
		current_samples_per_pass = samples_per_pass
		if samples_per_pass == -1:
			current_samples_per_pass = N_SPP

	left_samples = N_SPP
	completed_samples = 0
	n_pass = 0
	inv_variance_weights = []
	output_images = []
	final_image = np.zeros((height, width, 4))
	q_table_update_elapsed_time_accumulated = 0

	while left_samples > 0:
		context["samples_per_pass"] = np.array(current_samples_per_pass, dtype=np.uint32)
		print("Current Pass: %d, Current Samples: %d" % (n_pass, current_samples_per_pass))
		epsilon = getEpsilon(completed_samples, N_SPP, t=1)
		set_camera(context, scene.camera)
		context["frame_number"] = np.array((completed_samples + 1), dtype=np.uint32)
		context.launch(0, width, height)

		total_path_length_sum = np.sum(context['path_length_buffer'].to_array())
		visit_counts = context['visit_counts'].to_array()
		total_visited_counts_sum = np.sum(visit_counts)
		temp_difference = total_path_length_sum - total_visited_counts_sum
		print("path minus visited counts", temp_difference)

		if need_q_table_update:
			q_table_update_start_time = time.time()
			# print(visit_counts[0])
			# print(q_table[0])

			if accumulative_q_table_update:
				q_table_accumulated = context['q_table_accumulated'].to_array()
				q_table = np.divide(q_table_accumulated, visit_counts, out=np.zeros_like(q_table), where=visit_counts!=0.0)
				context['q_table'].copy_from_array(q_table)
			else:
				context["q_table"].copy_to_array(q_table)

			context['irradiance_table'] = Buffer.empty((n_a, n_s), dtype=np.float32, buffer_type='io', drop_last_dim=False)

			q_table += 1e-6
			q_table_sum = np.sum(q_table, axis=0, keepdims=True)
			policy_table = np.divide(q_table, q_table_sum)
			policy_table = policy_table * (1 - epsilon) + equal_table * epsilon
			context['q_table_old'].copy_from_array(policy_table)

			q_table_update_elapsed_time = time.time() - q_table_update_start_time
			q_table_update_elapsed_time_accumulated += q_table_update_elapsed_time
		np_hit_count = context['hit_count_buffer'].to_array()
		hit_new_sum = np.sum(np_hit_count)

		if n_pass > 0:
			hit = hit_new_sum - hit_sum
			hit_counts.append(hit)

		hit_sum = hit_new_sum

		output_image = context['output_buffer'].to_array()

		# if use inverse variance stack images
		if sample_combination == "inverse_variance":
			output2_image = context['output_buffer2'].to_array()
			variance = output2_image - output_image * output_image
			variance_lum = np.clip(get_luminance(variance), 1e-4, 10000)
			inv_variance = 1 / np.mean(variance_lum)
			# inv_variance = np.mean(1 / variance_lum)
			inv_variance_weights.append(inv_variance)
		elif sample_combination == "discard":
			final_image = output_image
		else:
			final_image += current_samples_per_pass * output_image

		if sample_combination == "inverse_variance" or export_video:
			output_images.append(output_image)

		# update next pass
		left_samples -= current_samples_per_pass
		completed_samples += current_samples_per_pass
		if learning_method == "exponential":
			current_samples_per_pass *= 2
			if left_samples - current_samples_per_pass < 2 * current_samples_per_pass:
				current_samples_per_pass = left_samples
		else:
			current_samples_per_pass = samples_per_pass
		current_samples_per_pass = min(current_samples_per_pass, left_samples)
		n_pass += 1

	if export_video:
		accumulated_image = None
		height, width, layers = output_images[0].shape
		out = cv2.VideoWriter('../video/%s_%s.avi' % (scene.name, export_video_name), cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))

		for i, image in enumerate(output_images):

			if i == 0:
				accumulated_image = np.copy(image)
			else:
				accumulated_image += image

			accumulated_image_avg = accumulated_image / (i + 1)
			accumulated_image_avg = np.flipud(accumulated_image_avg)
			accumulated_image_avg = accumulated_image_avg[:, :, 0:3]
			accumulated_image_avg = cv2.cvtColor(accumulated_image_avg, cv2.COLOR_RGB2BGR)
			accumulated_image_avg = LinearToSrgb(ToneMap(accumulated_image_avg, 1.5))
			accumulated_image_avg = convert_image_to_uint(accumulated_image_avg)
			# accumulated_images.append(accumulated_image_avg)
			out.write(accumulated_image_avg)
		out.release()
		print("Export finish!")

	if sample_combination == "inverse_variance":
		final_image = np.zeros_like(output_images[0])
		accumulated_weight = 0
		for _ in range(4):
			weight = inv_variance_weights.pop()
			print("weight", weight)
			final_image += weight * output_images.pop()
			accumulated_weight += weight
		final_image /= accumulated_weight
	elif sample_combination == "none":
		final_image = final_image / N_SPP

	print(final_image.shape)
	final_image = np.flipud(final_image)
	hdr_image = final_image[:, :, 0:3]
	#hdr_image = cv.cvtColor(hdr_image, cv.COLOR_RGB2BGR)
	#tone_map = cv.createTonemapReinhard(gamma=2.2, intensity=0, light_adapt=0, color_adapt=0)
	#ldr_image = tone_map.process(hdr_image)
	#ldr_image = cv.cvtColor(ldr_image, cv.COLOR_RGB2BGR)
	ldr_image = LinearToSrgb(ToneMap(hdr_image, 1.5))

	error_mean = 0.1
	if reference_image is not None:
		error = np.abs(ldr_image - reference_image)
		error_mean = np.mean(error)
		print("MAE:", error_mean)

	# if reference_image_hdr is not None:
	# 	error = np.abs(hdr_image - reference_image_hdr)
	# 	error_mean = np.mean(error)
	# 	print("HDR MAE:", error_mean)

	np_hit_count = context['hit_count_buffer'].to_array()
	total_hit_count = np.sum(np_hit_count)
	total_hit_percentage = total_hit_count / ((width * height) * N_SPP)
	print("Hit percent", total_hit_percentage)

	path_length = context['path_length_buffer'].to_array()
	total_path_length = np.sum(path_length)
	total_path_length_avg = total_path_length / ((width * height) * N_SPP)
	print("Hit path length avg", total_path_length_avg)

	print(np.mean(ldr_image), "Image MEAN")

	end_time = time.time()
	elasped_time = end_time - start_time
	elasped_time -= q_table_update_elapsed_time_accumulated

	print("Elapsed Time:", str(timedelta(seconds=elasped_time)))
	print("Accumulated time", str(timedelta(seconds=q_table_update_elapsed_time_accumulated)))
	elasped_time_per_spp = elasped_time / N_SPP

	if save_image:
		if save_image_name is None:
			file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
			save_pred_images(ldr_image, "../data/reference_images/reference_image_%s"% file_name)
		else:
			save_pred_images(ldr_image, "../data/reference_images/%s" % save_image_name)

	# import scipy.misc
	# scipy.misc.imsave('outfile.jpg', np_image[:,:,0])
	if show_picture:
		plt.imshow(ldr_image)
		plt.show()
		q_table2 = np.amax(q_table, axis=0)
		q_table2 = np.reshape(q_table2, [N_CUBE, N_CUBE, N_CUBE])

		if sample_type == SAMPLE_Q_PROPORTION and show_q_value_map:
			fig = plt.figure()

			for n_pass in range(N_CUBE):
				q_temp = q_table2[:, n_pass, :]
				ax1 = fig.add_subplot(4, 4, n_pass+1)
				ax1.imshow(q_temp, origin='lower')
			plt.show()

	results = dict()
	results["hit_count_sequence"] = hit_counts
	results["final_image"] = ldr_image
	results["error_mean"] = error_mean
	results["total_hit_percentage"] = total_hit_percentage
	results["elapsed_time_per_spp"] = elasped_time_per_spp
	return results


def set_sigmas(context):
	context["sigma_s"] = np.array(sigma_s, dtype=np.float32)
	context["sigma_a"] = np.array(sigma_a, dtype=np.float32)
	context["sigma_t"] = np.array(sigma_t, dtype=np.float32)
	context["hg_g"] = np.array(hg_g, dtype=np.float32)


def create_context():
	context = Context()

	context.set_ray_type_count(2)
	context.set_entry_point_count(1)
	context.set_stack_size(2000)

	context['scene_epsilon'] = np.array(1e-3, dtype=np.float32)
	context['pathtrace_ray_type'] = np.array(0, dtype=np.uint32)
	context['pathtrace_shadow_ray_type'] = np.array(1, dtype=np.uint32)
	context['rr_begin_depth'] = np.array(8, dtype=np.uint32)
	context['max_depth'] = np.array(16, dtype=np.uint32)

	Program(target_program, 'pathtrace_camera')
	entry_point = EntryPoint(Program(target_program, 'pathtrace_camera'),
							 Program(target_program, 'exception'),
							 Program(target_program, 'miss'))
	context['sqrt_num_samples'] = np.array(N_SQRT_NUM_SAMPLES, dtype=np.uint32)
	context['bad_color'] = np.array([1000000., 0., 1000000.], dtype=np.float32)
	context['bg_color'] = np.zeros(3, dtype=np.float32)

	return context, entry_point


def create_q_table_related(context, room_size):
	input_array = np.zeros((UV_N * UV_N * 2, 3), dtype=np.float32)
	for i in range(2 * UV_N * UV_N):
		v = getDirectionFrom(i, (0.5, 0.5), (UV_N, UV_N))
		input_array[i][0] = v[0]
		input_array[i][1] = v[1]
		input_array[i][2] = v[2]

	context['unitUVVectors'] = Buffer.from_array(input_array, dtype=np.float32, buffer_type='i', drop_last_dim=True)
	context['output_buffer'] = Buffer.empty((height, width, 4), dtype=np.float32, buffer_type='o', drop_last_dim=True)
	context['output_buffer2'] = Buffer.empty((height, width, 4), dtype=np.float32, buffer_type='o', drop_last_dim=True)
	context['hit_count_buffer'] = Buffer.empty((height, width, 1), dtype=np.float32, buffer_type='o',
											   drop_last_dim=True)
	context['path_length_buffer'] = Buffer.empty((height, width, 1), dtype=np.float32, buffer_type='o',
												 drop_last_dim=True)


	unit_cube_number = np.array([N_CUBE, N_CUBE, N_CUBE], dtype=np.uint32)
	unit_cube_size = room_size / unit_cube_number.astype(np.float32)

	sphere_uv_map_number = np.array([UV_N, UV_N], dtype=np.uint32)

	state_number = int(np.prod(unit_cube_number))
	action_number = int(np.prod(sphere_uv_map_number)) * 2

	print("Total State Number", state_number, action_number)

	q_table_init = np.zeros((action_number, state_number), dtype=np.float32)
	q_table_init.fill(1e-3)

	policy_table_init = np.zeros((action_number, state_number), dtype=np.float32)
	policy_table_init.fill(1e-3)

	normal_table_init = np.zeros((3, state_number), dtype=np.float32)
	normal_table_init.fill(1e-3)
	normal_table_init2 = np.zeros((3, state_number), dtype=np.float32)
	normal_table_init2.fill(1e-3)

	visit_counts = np.zeros((action_number, state_number), dtype=np.uint32)
	mcmc_init = np.random.random((action_number, state_number, 2)).astype(np.float32)
	print(mcmc_init.dtype)
	context['q_table_accumulated'] = Buffer.empty((action_number, state_number), dtype=np.float32, buffer_type='io', drop_last_dim=False)
	context['irradiance_table'] = Buffer.empty((action_number, state_number), dtype=np.float32, buffer_type='io', drop_last_dim=False)

	context['q_table'] = Buffer.from_array(q_table_init, dtype=np.float32, buffer_type='io', drop_last_dim=False)
	context['q_table_old'] = Buffer.from_array(policy_table_init, dtype=np.float32, buffer_type='io', drop_last_dim=False)
	context['normal_table'] = Buffer.from_array(normal_table_init, dtype=np.float32, buffer_type='io', drop_last_dim=False)
	context['normal_table_old'] = Buffer.from_array(normal_table_init2, dtype=np.float32, buffer_type='io', drop_last_dim=False)
	context['mcmc_table'] = Buffer.from_array(mcmc_init, dtype=np.float32, buffer_type='io', drop_last_dim=True)

	context['visit_counts'] = Buffer.from_array(visit_counts, dtype=np.float32, buffer_type='io', drop_last_dim=False)

	context['unitCubeSize'] = unit_cube_size
	context['unitCubeNumber'] = unit_cube_number
	context['unitUVNumber'] = sphere_uv_map_number

	return q_table_init


def register_bsdf_programs(context):
	BRDFEval = np.zeros([1, ], dtype=np.int32)
	BRDFEval[0] = Program('optix/bsdf/lambert.cu', 'Eval').get_id()
	print("Eval programs", BRDFEval)

	BRDFSample = np.zeros([1, ], dtype=np.int32)
	BRDFSample[0] = Program('optix/bsdf/lambert.cu', 'Sample').get_id()
	print("Sample programs", BRDFSample)

	BRDFPdf = np.zeros([1, ], dtype=np.int32)
	BRDFPdf[0] = Program('optix/bsdf/lambert.cu', 'Pdf').get_id()
	print("Pdf programs", BRDFPdf)

	context["sysBRDFPdf"] = Buffer.from_array(BRDFPdf, dtype=np.int32, buffer_type='i')
	context["sysBRDFSample"] = Buffer.from_array(BRDFSample, dtype=np.int32, buffer_type='i')
	context["sysBRDFEval"] = Buffer.from_array(BRDFEval, dtype=np.int32, buffer_type='i')


def create_scene_lights(context, scene):
	lights = []
	for light_data in scene.lights:
		light = Light(light_data)
		print("- Light Data: ", light_data)
		lights.append(np.array(light))
	np_l = np.array(lights)
	light_buffer = Buffer.from_array(np_l, dtype=Light.dtype, buffer_type='i', drop_last_dim=True)
	context["lights"] = light_buffer


def create_geometry(context, scene):
	geometry_instances = scene.geometry_instances
	light_instances = scene.light_instances

	shadow_group = Group(children=geometry_instances)
	shadow_group.set_acceleration(Acceleration("Trbvh"))
	context['top_shadower'] = shadow_group

	group = Group(children=(geometry_instances + light_instances))
	group.set_acceleration(Acceleration("Trbvh"))
	context['top_object'] = group


def normalize(mat):
	return mat / np.linalg.norm(mat)

from collections import OrderedDict


def show_result_graph(dictionaries, target_key):
	plt.figure()
	names = []
	values = []
	for k, v in dictionaries.items():
		name = k
		names.append(name)
		values.append(v[target_key])

	plt.bar(names, values)
	value_set = set(values)
	if len(value_set) > 1:
		low = min(values)
		high = max(values)
		plt.ylim([max((low - 0.5 * (high - low)), 0), (high + 0.5 * (high - low))])
	plt.xticks(rotation=90)
	plt.ylabel(target_key)
	plt.show()

def show_hit_percentage_sequence(dictionaries):
	plt.figure()
	names = []
	for k, v in dictionaries.items():
		name = k
		names.append(name)
		N = len(v["hit_count_sequence"])
		plt.plot([_ for _ in range(N)], v["hit_count_sequence"], label=k)

	# plt.bar(names, values)
	# value_set = set(values)
	# if len(value_set) > 1:
	# 	low = min(values)
	# 	high = max(values)
	# 	plt.ylim([max((low - 0.5 * (high - low)), 0), (high + 0.5 * (high - low))])
	# plt.xticks(rotation=90)
	# plt.ylabel(target_key)
	plt.legend()
	plt.show()


def load_reference_image(name):
	global reference_image, reference_image_hdr
	# try:
	# 	reference_image_hdr = cv.imread("../data/reference_images/%s.exr" % name)
	# 	reference_image_hdr = cv.cvtColor(reference_image_hdr, cv.COLOR_RGB2BGR)
	# 	reference_image_hdr = np.asarray(reference_image_hdr, dtype=np.float32)
	# 	print("EXR ref image shape", reference_image_hdr.shape)
	# except IOError:
	# 	pass
	#
	# reference_image = cv.imread("../data/reference_images/%s.png" % name)
	# reference_image = cv.cvtColor(reference_image, cv.COLOR_RGB2BGR)
	# reference_image = np.asarray(reference_image, dtype=np.float32)
	# reference_image /= 255.0

	reference_image = Image.open("../data/reference_images/%s.png" % name)
	reference_image = np.asarray(reference_image, dtype=np.float32)
	reference_image = reference_image[:,:,0:3]
	reference_image /= 255.0
	print("PNG ref image shape", reference_image.shape)


def make_reference_image():
	render(SAMPLE_COSINE, show_picture=True, scatter_sample_type=SAMPLE_HG, save_image=True, use_mis=False)


def set_scene(scene_name_target, sample_sqrt, _sigma_s = 0.0, _sigma_a=0.0, _hg_g=0.9):
	global scene_name, N_SQRT_NUM_SAMPLES, N_SPP, sigma_s, sigma_a, sigma_t, hg_g
	scene_name = scene_name_target
	N_SQRT_NUM_SAMPLES = sample_sqrt
	N_SPP = sample_sqrt * sample_sqrt
	sigma_s = _sigma_s
	sigma_a = _sigma_a
	sigma_t = sigma_s + sigma_a
	hg_g = _hg_g



def run():
	load_reference_image("cornell_box_16384")
	#
	total_results = OrderedDict()
	# total_results["qcos_hg"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=False, scatter_sample_type=SAMPLE_HG)
	# total_results["q_hg"] = render(SAMPLE_Q_PROPORTION, show_picture=False, scatter_sample_type=SAMPLE_HG)
	#
	# total_results["uniform_no_mis"] = render(SAMPLE_UNIFORM, show_picture=False, scatter_sample_type=SAMPLE_HG, save_image=False, use_mis=False)
	total_results["cosine_no_mis_hg"] = render(SAMPLE_COSINE, show_picture=True, scatter_sample_type=SAMPLE_HG, save_image=False, use_mis=False)
	total_results["cosine_no_mis_qhg"] = render(SAMPLE_COSINE, show_picture=True, scatter_sample_type=SAMPLE_Q_HG_PROPORTION, save_image=False, use_mis=False)


	# total_results["uniform_mis"] = render(SAMPLE_UNIFORM, show_picture=False, scatter_sample_type=SAMPLE_HG, save_image=False, use_mis=True)
	# total_results["cosine_mis"] = render(SAMPLE_COSINE, show_picture=True, scatter_sample_type=SAMPLE_HG, save_image=False, use_mis=True)
	# total_results["q_mis"] = render(SAMPLE_Q_PROPORTION, show_picture=False, scatter_sample_type=SAMPLE_HG, save_image=False, use_mis=True)
	# total_results["qcosine_mis"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=True, scatter_sample_type=SAMPLE_HG, save_image=False, use_mis=False)
	# total_results["qcosine_no_mis"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=True, scatter_sample_type=SAMPLE_HG, save_image=False, use_mis=False)


	#show_result_graph(total_results, "error_mean")
	#show_result_graph(total_results, "total_hit_percentage")
	#show_result_graph(total_results, "elapsed_time_per_spp")




