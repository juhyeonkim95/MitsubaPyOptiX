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


ESCAPE_KEY = 27

width = 512
height = 512
Compiler.add_program_directory(dirname(__file__))
target_program = 'optix/integrators/optixPathTracerQTable.cu'
N_SQRT_NUM_SAMPLES = 16
N_SPP = N_SQRT_NUM_SAMPLES * N_SQRT_NUM_SAMPLES
N_CUBE = 16
UV_N = 6

SAMPLE_UNIFORM = 0
SAMPLE_COSINE = 1
SAMPLE_HG = 1
SAMPLE_Q_PROPORTION = 2
SAMPLE_Q_COS_PROPORTION = 3
SAMPLE_Q_HG_PROPORTION = 3


Q_UPDATE_EXPECTED_SARSA = 0
Q_UPDATE_Q_LEARNING = 1
Q_UPDATE_SARSA = 2

Q_SAMPLE_PROPORTIONAL_TO_Q = 1
Q_SAMPLE_PROPORTIONAL_TO_Q_SQUARE = 2


use_upside_down_light = False
reference_image = None


def save_pred_images(images, file_path):
	x = np.copy(images)
	x *= 255
	x = np.clip(x, 0, 255)
	x = x.astype('uint8')
	new_im = Image.fromarray(x)
	new_im = new_im.transpose(Image.FLIP_TOP_BOTTOM)
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



class ParallelogramLight:
	def __init__(self):
		# structure : [corner, v1, v2, normal, emission] (all float3)
		self.buffer_numpy = np.zeros(15, dtype='<f4')

	@property
	def buffer(self):
		return self.buffer_numpy.tobytes()

	def set_corner(self, x, y, z):
		self.buffer_numpy[0] = x
		self.buffer_numpy[1] = y
		self.buffer_numpy[2] = z

	def set_v1_v2(self, v1, v2):
		normal = np.cross(v1, v2)
		normal /= np.linalg.norm(normal)
		self.buffer_numpy[3:6] = v1
		self.buffer_numpy[6:9] = v2
		self.buffer_numpy[9:12] = normal

	def set_emission(self, r, g, b):
		self.buffer_numpy[12] = r
		self.buffer_numpy[13] = g
		self.buffer_numpy[14] = b


def render(
		sample_type=0,
		q_table_update_method=Q_UPDATE_EXPECTED_SARSA,
		q_value_sample_method=1,
		q_value_sample_constant=1.0,
		show_q_value_map=False,
		show_picture=True,
		save_image=False,
		scatter_sample_type=1
		):

	start_time = time.time()
	context, entry_point = create_context()
	context["sample_type"] = np.array(sample_type, dtype=np.uint32)
	context["scatter_sample_type"] = np.array(scatter_sample_type, dtype=np.uint32)

	context["q_table_update_method"] = np.array(q_table_update_method, dtype=np.uint32)
	context["q_value_sample_method"] = np.array(q_value_sample_method, dtype=np.uint32)
	context["q_value_sample_constant"] = np.array(q_value_sample_constant, dtype=np.float32)

	# context.set_print_enabled(True)
	# context.set_print_buffer_size(20000)
	scene_name = "cornell_box"
	scene = Scene(scene_name)
	scene.load_scene_from("../data/%s/scene.xml" % scene_name)
	scene.load_images()
	scene.create_objs()
	scene.create_object_instances()

	room_size = scene.bbox.bbox_max - scene.bbox.bbox_min
	print("Room Size:", room_size)
	q_table_original = create_q_table_related(context, room_size)

	create_geometry(context, scene)

	[n_a, n_s] = q_table_original.shape
	#h_n_a = n_a // 2
	q_table = np.zeros((n_a, n_s), dtype=np.float32)
	#print(np.sum(q_table))
	entry_point.launch((width, height))

	equal_table = np.zeros((n_a, n_s), dtype=np.float32)
	equal_table.fill(1 / n_a)
	epsilon = 1
	hit_sum = 0
	hit_counts = []
	for i in range(N_SPP):
		print(i)
		epsilon = getEpsilon(i, N_SPP, t=1)

		# print(i, epsilon)#1 - (i+1) / (N_SQRT_NUM_SAMPLES * N_SQRT_NUM_SAMPLES)
		#window.set_camera(scene.camera)
		set_camera(context, scene.camera)
		context["frame_number"] = np.array((i + 1), dtype=np.uint32)
		context.launch(0, width, height)

		if sample_type == SAMPLE_Q_PROPORTION \
				or sample_type == SAMPLE_Q_COS_PROPORTION \
				or scatter_sample_type == SAMPLE_Q_PROPORTION\
				or scatter_sample_type == SAMPLE_Q_HG_PROPORTION:
			context["q_table"].copy_to_array(q_table)
			#print(q_table.sum(), "qtable sum")
			# q_table = context['q_table'].to_array()
			# print(q_table.shape, "Q table Shape")
			q_table_sum = np.sum(q_table, axis=0, keepdims=True)
			#print(q_table_sum.sum())
			policy_table = np.divide(q_table, q_table_sum)
			policy_table = policy_table * (1 - epsilon) + equal_table * (epsilon)
			#policy_table.fill(1/n_a)
			context['q_table_old'].copy_from_array(policy_table)
		np_hit_count = context['hit_count_buffer'].to_array()
		hit_new_sum = np.sum(np_hit_count)

		if i>0:
			hit = hit_new_sum - hit_sum
			hit_counts.append(hit)

		#print("HIT COUNT", hit_new_sum - hit_sum)
		#print("Q VALUE SUM", np.sum(q_table) - n_s * n_a)
		hit_sum = hit_new_sum

	np_image = context['output_buffer'].to_array()
	np_image = np_image / N_SPP

	if reference_image is not None:
		error = np_image - reference_image
		error = np.abs(error[:, :, 0:3])
		error_mean = np.mean(error)
		print("MAE:", error_mean)


	np_image2 = context['output_buffer2'].to_array()
	np_image2 = np_image2 / N_SPP
	variance = np_image2 - np_image * np_image

	variance /= (np_image + 0.0001)

	variance_val_max = np.amax(variance)
	variance /= variance_val_max

	print(variance.sum(), "variance sum")

	np_hit_count = context['hit_count_buffer'].to_array()
	#print(np.amax(np_hit_count), np.amin(np_hit_count), "MAX MIN")
	total_hit_count = np.sum(np_hit_count)
	total_hit_percentage = total_hit_count / ((width * height) * N_SPP)
	print("Hit percent", total_hit_percentage)

	path_length = context['path_length_buffer'].to_array()
	#print(np.amax(path_length), np.amin(path_length), "MAX MIN")
	#path_length = path_length / (N_SQRT_NUM_SAMPLES * N_SQRT_NUM_SAMPLES)
	total_path_length = np.sum(path_length)
	total_path_length_avg = total_path_length / ((width * height) * N_SPP)
	print("Hit path length avg", total_path_length_avg)

	print(np.mean(np_image), "Image MEAN")

	end_time = time.time()
	print("Elapsed Time:", str(timedelta(seconds=end_time - start_time)))

	if save_image:
		file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		save_pred_images(np_image, "reference_image_%s"% file_name)

	# import scipy.misc
	# scipy.misc.imsave('outfile.jpg', np_image[:,:,0])
	if show_picture:
		plt.imshow(np_hit_count[:,:,0] / N_SPP, origin='lower')
		plt.show()
		plt.figure()
		plt.imshow(np_image, origin='lower')
		plt.show()

		#plt.figure()
		#plt.imshow(variance[:, :, 0:3], origin='lower')
		#plt.show()

		q_table2 = np.amax(q_table, axis=0)
		q_table2 = np.reshape(q_table2, [N_CUBE, N_CUBE, N_CUBE])

		if sample_type == SAMPLE_Q_PROPORTION and show_q_value_map:
			fig = plt.figure()

			for i in range(N_CUBE):
				#
				q_temp = q_table2[:, i, :]
				#print(i, "AAA")
				#print(q_temp)
				ax1 = fig.add_subplot(4, 4, i+1)

				ax1.imshow(q_temp, origin='lower')
			plt.show()

	results = dict()
	results["hit_count_sequence"] = hit_counts
	results["final_image"] = np_image
	results["error_mean"] = 0
	results["total_hit_percentage"] = total_hit_percentage

	return results

	# print(np.amax(np_image))
	# print(np.amin(np_image))
	# print(np_image[100,100:120,0])
	# print(np_image.shape)
	# from PIL import Image
	# im = Image.fromarray(np_image, 'RGBA')
	# im.save("image.png")

	#window.run()


def create_context():
	context = Context()

	context.set_ray_type_count(2)
	context.set_entry_point_count(1)
	context.set_stack_size(2000)

	context['scene_epsilon'] = np.array(1e-3, dtype=np.float32)
	context['pathtrace_ray_type'] = np.array(0, dtype=np.uint32)
	context['pathtrace_shadow_ray_type'] = np.array(1, dtype=np.uint32)
	context['rr_begin_depth'] = np.array(10, dtype=np.uint32)

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

	# room_size = np.array([560, 560, 560], dtype=np.float32)
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

	visit_counts = np.zeros((action_number, state_number), dtype=np.uint32)

	# context['q_table'] = Buffer.empty((state_number, action_number), dtype=np.float32, buffer_type='io')
	context['q_table'] = Buffer.from_array(q_table_init, dtype=np.float32, buffer_type='io', drop_last_dim=False)
	context['q_table_old'] = Buffer.from_array(policy_table_init, dtype=np.float32, buffer_type='io',
											   drop_last_dim=False)

	context['visit_counts'] = Buffer.from_array(visit_counts, dtype=np.float32, buffer_type='io', drop_last_dim=False)

	context['unitCubeSize'] = unit_cube_size
	context['unitCubeNumber'] = unit_cube_number
	context['unitUVNumber'] = sphere_uv_map_number

	return q_table_init


def create_geometry_instance(geometry, material, variable=None, color=None):
	geometry_instance = GeometryInstance(geometry, material)
	if variable is not None:
		geometry_instance[variable] = color

	return geometry_instance


def create_glass_material():
	glass = Material(closest_hit={0: Program(target_program, 'glass')})
	glass["refraction_index"] = np.array([1.4], dtype=np.float32)
	glass["refraction_color"] = np.array([0.99, 0.99, 0.99], dtype=np.float32)
	glass["reflection_color"] = np.array([0.99, 0.99, 0.99], dtype=np.float32)
	glass["extinction"] = np.array([0,0,0], dtype=np.float32)
	return glass

def create_geometry(context, scene):
	light = ParallelogramLight()
	light.set_corner(343., 548.6, 227.)
	light.set_v1_v2(np.array([-130., 0., 0.]), np.array([0., 0., 105.]))
	light.set_emission(15, 15, 15)
	light_buffer = Buffer.from_array([light.buffer_numpy.tobytes()], buffer_type='i')
	context["lights"] = light_buffer

	geometry_instances = scene.geometry_instances
	light_instances = scene.light_instances

	shadow_group = GeometryGroup(children=geometry_instances)
	shadow_group.set_acceleration(Acceleration("Trbvh"))
	context['top_shadower'] = shadow_group

	group = GeometryGroup(children=(geometry_instances + light_instances))
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
	low = min(values)
	high = max(values)
	plt.ylim([max((low - 0.5 * (high - low)), 0), (high + 0.5 * (high - low))])
	plt.xticks(rotation=90)
	plt.ylabel(target_key)
	plt.show()


def run():
	global reference_image

	# plt.figure()

	# reference_image = Image.open("reference_image.png")
	# reference_image = reference_image.transpose(Image.FLIP_TOP_BOTTOM)
	# reference_image = np.array(reference_image, dtype=np.float32)
	# reference_image /= 255.0

	total_results = OrderedDict()
	# total_results["qcos_hg"] = render(SAMPLE_Q_COS_PROPORTION, show_picture=True, scatter_sample_type=SAMPLE_HG)
	total_results["cosine_hg"] = render(SAMPLE_COSINE, show_picture=True, scatter_sample_type=SAMPLE_HG, save_image=False)


if __name__ == '__main__':
	run()

