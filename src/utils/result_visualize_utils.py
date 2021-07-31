from utils.image_utils import *
from utils.result_export_utils import *


def show_result_bar(dictionaries, target_key):
	plt.figure()
	names = []
	values = []
	for k, v in dictionaries.items():
		name = k
		names.append(name)
		values.append(v[target_key])

	plt.bar(names, values)
	value_set = set(values)
	print(values)
	if len(value_set) > 1:
		low = min(values)
		high = max(values)
		print(low, high)
		plt.ylim([max((low - 0.5 * (high - low)), 0), (high + 0.5 * (high - low))])
	plt.xticks(rotation=90)
	plt.ylabel(target_key)
	plt.show()


def show_result_stacked_bar(dictionaries, target_key1, target_key2):
	plt.figure()
	names = []
	values1 = []
	values2 = []

	for k, v in dictionaries.items():
		name = k
		names.append(name)
		values1.append(v[target_key1])
		values2.append(v[target_key2])

	fig, ax = plt.subplots()

	ax.bar(names, values1, label=target_key1)
	ax.bar(names, values2, label=target_key2)

	ax.tick_params(labelrotation=90)
	ax.set_ylabel("sec")
	ax.legend()
	plt.show()


def show_result_sequence(dictionaries, key):
	plt.figure()
	names = []
	for k, v in dictionaries.items():
		name = k
		names.append(name)
		N = len(v[key])
		plt.plot([_ for _ in range(N)], v[key], label=k)
	plt.ylabel(key)
	plt.legend()
	plt.show()


def show_result_func(total_results):
	show_result_bar(total_results, "error_mean")
	show_result_bar(total_results, "elapsed_time_per_sample")
	show_result_bar(total_results, "elapsed_time_per_sample_except_init")
	show_result_stacked_bar(total_results, "total_optix_launch_time", "total_q_table_update_time")

	show_result_bar(total_results, "total_hit_percentage")
	show_result_bar(total_results, "completed_samples")

	# sequence
	show_result_sequence(total_results, "hit_rate_per_pass")
	show_result_sequence(total_results, "elapsed_time_per_sample_per_pass")
	show_result_sequence(total_results, "q_table_update_times")