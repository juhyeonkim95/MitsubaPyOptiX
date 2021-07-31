# from renderer import *
# import matplotlib.pyplot as plt
# import datetime
#
#
# def test():
# 	renderer = Renderer()
# 	#scene_name = "material-testball"
# 	scene_name = "bathroom"
# 	image = renderer.render(scene_name, _spp=256, use_mis=False)
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

from test_utils import *





def make_reference_image_single(scene_name, scale=4, force_all_diffuse=False):
    renderer = Renderer(scale=scale, force_all_diffuse=force_all_diffuse)
    common_params = {
        'scene_name': scene_name,
        '_spp': 1024,
        'samples_per_pass': 64,
        'max_depth': 8,
        'rr_begin_depth': 8,

        # You should change q_table_old at getQValue to q_table
        'accumulative_q_table_update': True
    }
    image = renderer.render(**common_params, use_mis=False, show_picture=True)
    file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# save_pred_images(image['image'], "../reference_images2/%s_%s" % (scene_name, file_name))


def export_radiance(scene_name, scale=4, n_uvs=None, render_reference=True, export=False):
    reference_parent_folder = '../reference_images/%s/scale_%d' % ("standard", scale)
    ref_image = load_reference_image(reference_parent_folder, scene_name)

    renderer = Renderer(scale=scale, force_all_diffuse=False)
    renderer.reference_image = ref_image
    n_cube = 8
    coord = (2, 7, 4)
    common_params = {'scene_name': scene_name, 'samples_per_pass': 8, 'show_picture': False, 'max_depth': 16,
                     'rr_begin_depth': 8, 'scene_epsilon': 1e-5, 'accumulative_q_table_update': True,
                     'n_cube': n_cube, '_spp': 1024*16, 'time_limit_init_ignore_step':0}

    pos = np.array(coord, dtype=np.float32) / n_cube
    size = np.array([1 / n_cube] * 3, dtype=np.float32)
    total_result = OrderedDict()

    if render_reference:
        total_result["ground_truth"] = renderer.render(**common_params, spherical_pos=pos, spherical_size=size, spherical_map_type=0, convert_ldr=False)
        radiance_field_gt = total_result["ground_truth"]['image']
        radiance_field_gt = radiance_field_gt.mean(axis=2)
        radiance_field_gt = np.transpose(radiance_field_gt)
        radiance_field_gt = radiance_field_gt[256:512, :]
        radiance_field_gt_image = Image.fromarray(radiance_field_gt)
        fig = plt.figure()
        plt.imshow(radiance_field_gt)
        plt.axis('off')
        fig.savefig('../result_radiance_examples/reference_radiance.png', bbox_inches='tight', pad_inches=0)
        np.save('../result_radiance_examples/reference_radiance', radiance_field_gt)
        plt.show()
    else:
        radiance_field_gt = np.load('../result_radiance_examples/reference_radiance.npy')
        radiance_field_gt_image = Image.fromarray(radiance_field_gt)
        fig = plt.figure()
        plt.imshow(radiance_field_gt)
        plt.axis('off')
        fig.savefig('../result_radiance_examples/reference_radiance.png', bbox_inches='tight', pad_inches=0)
        plt.show()

    total_result_to_export = OrderedDict()

    if n_uvs is None:
        n_uvs = [8]
    keys = ["expected_sarsa", "monte_carlo", "sarsa", "sarsa2"]
    target_exports = ["_time", "mean_absolute_error", "mean_error", "variance"]

    for t in target_exports:
        total_result_to_export[t] = OrderedDict()
        for key in keys:
            total_result_to_export[t][key] = []

    for n_uv in n_uvs:
        # renderer.render(**common_params, spherical_pos=pos, spherical_size=size, spherical_map_type=1)
        common_params["_spp"] = 1024
        common_params["uv_n"] = n_uv
        total_result["expected_sarsa"] = renderer.render(**common_params,
                                                         sampling_strategy=SAMPLE_COSINE,
                                                         force_update_q_table=True,
                                                         q_table_update_method=SAMPLE_COSINE)
        total_result["monte_carlo"] = renderer.render(**common_params,
                                                      sampling_strategy=SAMPLE_COSINE,
                                                      force_update_q_table=True,
                                                      q_table_update_method=Q_UPDATE_MONTE_CARLO)
        total_result["sarsa"] = renderer.render(**common_params,
                                                sampling_strategy=SAMPLE_COSINE,
                                                force_update_q_table=True,
                                                q_table_update_method=Q_UPDATE_SARSA)
        total_result["sarsa2"] = renderer.render(**common_params,
                                                 sampling_strategy=SAMPLE_COSINE,
                                                 force_update_q_table=True,
                                                 q_table_update_method=Q_UPDATE_SARSA2)

        # result = renderer.render(**common_params,
        #                          sampling_strategy=SAMPLE_COSINE,
        #                          force_update_q_table=True,
        #                          directional_mapping_method="cylindrical",
        #                          directional_type="quadtree",
        #                          learning_method="exponential",
        #                          q_table_update_method=Q_UPDATE_MONTE_CARLO)
        #radiance_expected_sarsa = total_result["expected_sarsa"]['q_table_info'].visualize_q_table(coord)
        #radiance_monte_carlo = total_result["monte_carlo"]['q_table_info'].visualize_q_table(coord)
        #radiance_sarsa = total_result["sarsa"]['q_table_info'].visualize_q_table(coord)

        # def calc_error(field):
        #     field_image = Image.fromarray(field)
        #     print("Shape", field.shape, radiance_field_gt.shape, field_image.size)
        #     field_image = field_image.resize((radiance_field_gt.shape))
        #     error = field_image - radiance_field_gt_image
        #     return error
        for key in keys:
            field = total_result[key]['q_table_info'].visualize_q_table(coord)
            field_width, field_height = field.shape
            field = field[field_width//2:field_width, :]

            field_image = Image.fromarray(field)
            field_image = field_image.resize(radiance_field_gt_image.size, Image.NEAREST)
            field_resized = np.asarray(field_image)

            error = field_resized - radiance_field_gt
            pdf_error = field_resized / np.sum(field_resized) - radiance_field_gt / np.sum(radiance_field_gt)
            rel_error = np.divide(error, radiance_field_gt, out=np.zeros_like(radiance_field_gt),
                                                          where=radiance_field_gt != 0.0)
            error = rel_error
            mean_error = np.mean(error)
            mean_absolute_error = np.abs(error).mean()
            error_variance = np.std(error)

            fig = plt.figure()
            plt.imshow(np.asarray(field_image), vmin=np.min(radiance_field_gt), vmax=np.max(radiance_field_gt))
            # plt.imshow(error)

            plt.axis('off')
            plt.show()
            print(key, mean_absolute_error)
            elapsed_time_per_sample = total_result[key]['elapsed_time_per_sample_except_init']

            total_result_to_export["mean_absolute_error"][key].append(mean_absolute_error)
            total_result_to_export["mean_error"][key].append(mean_error)
            total_result_to_export["variance"][key].append(error_variance)
            total_result_to_export["_time"][key].append(elapsed_time_per_sample)

            if export:
                fig.savefig('../result_radiance_examples/%s_uv_%d_radiance.png' % (key, n_uv), bbox_inches='tight', pad_inches=0)
    if export:
        for t in target_exports:
            df = pd.DataFrame(total_result_to_export[t], index=n_uvs)
            df.to_csv('../result_radiance_examples/%s.csv' % t)


# coo = result["q_table_visit_count_final"].reshape((n_cube, n_cube, n_cube, -1))
# coo = np.sum(coo, axis=3)
# print(coo.shape)
# print(np.unravel_index(coo.argmax(), coo.shape))
# print(np.argmax(coo))
#
# def visualize(name="q_table_final"):
# 	q_table = result[name]
# 	q_table = q_table.reshape((n_cube, n_cube, n_cube, n_uv * 2, n_uv))
# 	plt.figure()
# 	A = q_table[coord[2], coord[1], coord[0], :]
# 	print(A)
# 	print(A.shape)
# 	plt.imshow(A)
# 	plt.show()
#
# visualize("q_table_final")
# visualize("q_table_visit_count_final")





def test2(scene_name, scale=4, test_time=False, show_picture=False, show_result=False,
          output_folder=None, force_all_diffuse=False, sample_type_name=None, **kwargs):
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
        time_limit_in_secs = {1: 60, 2: 20, 4: 5}
        common_params['time_limit_in_sec'] = time_limit_in_secs[scale]
    else:
        common_params['_spp'] = 1024
    common_params['time_limit_init_ignore_step'] = 10

    total_results["uniform"] = renderer.render(**common_params, sampling_strategy=SAMPLE_UNIFORM)
    total_results["brdf"] = renderer.render(**common_params, sampling_strategy=SAMPLE_COSINE)
    n_cubes = [16, 14, 12, 10, 8, 6, 4]
    n_uvs = [16, 14, 12, 10, 8, 6, 4]
    print(kwargs)
    for n_c in n_cubes:
        for uv in n_uvs:
            total_results["%s_c_%d_uv_%d" % (sample_type_name, n_c, uv)] \
                = renderer.render(**common_params, uv_n=uv, n_cube=n_c, **kwargs)

    if show_result:
        show_result_bar(total_results, "error_mean")
        show_result_bar(total_results, "elapsed_time_per_sample")
        show_result_bar(total_results, "elapsed_time_per_sample_except_init")

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


if __name__ == '__main2__':
    # Compiler.clean()
    # Compiler.keep_device_function = False
    # file_dir = os.path.dirname(os.path.abspath(__file__))
    # Compiler.add_program_directory(file_dir)
    file_dir = os.path.dirname(os.path.abspath(__file__))
    Compiler.add_program_directory(file_dir)
    renderer = Renderer(scale=4)
    renderer.construct_stree('cornell-box', max_path_depth=2, max_octree_depth=6)

# if __name__ == '__main__':
# 	b = np.zeros((3,3))
# 	a = b[1]
# 	b = np.ones((3,3))
# 	print(a)
# 	print(b)

if __name__ == '__main__':
    Compiler.clean()
    Compiler.keep_device_function = False

    file_dir = os.path.dirname(os.path.abspath(__file__))
    Compiler.add_program_directory(file_dir)
    # make_reference_image_single("cornell-box", scale=2, force_all_diffuse=False)
    # test2("cornell-box", scale=4, force_all_diffuse=False, show_picture=False, show_result=True)

    # make_reference_image_single("veach-ajar", scale=4, force_all_diffuse=True)

    # make_reference_image_single("veach_door_simple", scale=2)
    # make_reference_image_multiple(scale=4, scene_names=["cornell-box-hard"])
    all_scenes1 = ['cornell-box', 'cornell-box-hard', "veach_door_simple"]
    all_scenes = all_scenes1 + ['bathroom', 'bathroom2', 'bedroom',  'kitchen',
                  'living-room', 'living-room-2', 'living-room-3', 'staircase', 'staircase2',
                  'veach-ajar']
    #make_reference_image_multiple(scale=4, scene_names=all_scenes)

    total_dict = OrderedDict()
    #update_total_result("../result_0411_1/scale_4_time_5", test_time=True)
    #test_multiple_and_export_result(all_scenes, 4, "../result_0412_1/scale_4_time_5", test_time=True, _time=5)
    #test_multiple_and_export_result(all_scenes, 4, "../result_0412_1/scale_4_spp_256", test_time=False, _spp=256)
    #test_multiple_and_export_result(all_scenes, 4, "../result_0412_1/scale_4_time_10", test_time=True, _time=10)
    #test_multiple_and_export_result(all_scenes, 4, "../result_0412_1/scale_4_spp_1024", test_time=False, _spp=1024)

    #test_multiple_and_export_result(all_scenes, 2, "../result_0412_3/scale_2_time_20", test_time=True, _time=20)
    #test_multiple_and_export_result(all_scenes, 2, "../result_0412_3/scale_2_spp_256", test_time=False, _spp=256)
    #test_multiple_and_export_result(all_scenes, 2, "../result_0412_3/scale_2_time_40", test_time=True, _time=40)
    #test_multiple_and_export_result(all_scenes, 2, "../result_0412_3/scale_2_spp_1024", test_time=False, _spp=1024)
    #test_multiple_and_export_result(all_scenes, 4, "../result_0412_3/scale_4_spp_1024", test_time=False, _spp=1024)

    #test_multiple_and_export_result(all_scenes, 4, "../result_0414_compare_epsilon_opt/scale_4_time_5", test_time=True, _time=5)
    #test_multiple_and_export_result(all_scenes, 4, "../result_0414_compare_epsilon_opt/scale_4_time_10", test_time=True, _time=10)
    #test_multiple_and_export_result(all_scenes, 2, "../result_0414_compare_epsilon_opt/scale_2_time_40", test_time=True, _time=40)
    #test_multiple_and_export_result(all_scenes, 2, "../result_0414_compare_epsilon_opt/scale_2_time_20", test_time=True, _time=20)
    #test_multiple_and_export_result(all_scenes, 2, "../result_0414_2_compare_epsilons/scale_2_time_40", test_time=True, _time=40)

    # test_multiple_and_export_result(all_scenes, 4, "../result_rejection_quad_tree_test/scale_4_time_10", test_time=True, _time=10)
    # test_multiple_and_export_result(all_scenes, 2, "../result_0414_2_compare_epsilons/scale_2_time_40", test_time=True, _time=40)
    #test_multiple_and_export_result(all_scenes, 4, "../result_memoization_compare/scale_4_time_5", test_time=True, _time=5)
    #test_multiple_and_export_result(all_scenes, 4, "../result_memoization_compare/scale_4_spp_256", test_time=False, _spp=256)
    #test_multiple_and_export_result(all_scenes, 4, "../result_memoization_compare/scale_4_spp_256", test_time=False, _spp=256)
    #test_multiple_and_export_result(all_scenes, 2, "../result_memoization_compare/scale_2_time_40", test_time=True, _time=40)
    #test_multiple_and_export_result(all_scenes, 2, "../result_memoization_compare/scale_2_spp_256", test_time=False, _spp=256)

    #test_multiple_and_export_result(all_scenes, 2, "../result_quadtree_again/scale_2_time_40", test_time=True, _time=40)

    # test("cornell-box", 4, test_time=True, show_picture=False, show_result=True, _time=5,
    #      sampling_strategy=SAMPLE_Q_COS_REJECT, update_type=Q_UPDATE_MONTE_CARLO)
    #test("cornell-box", 4, test_time=True, show_picture=False, show_result=True, _time=5)
    #test("cornell-box", 4, test_time=True, show_picture=False, show_result=True, _time=5,
    #     sampling_strategy=SAMPLE_Q_COS_REJECT, update_type=Q_UPDATE_Q_LEARNING)
    # test("cornell-box", 4, test_time=True, show_picture=False, show_result=True, _time=5,
    #      sampling_strategy=SAMPLE_Q_COS_REJECT, update_type=Q_UPDATE_EXPECTED_SARSA)
    # test("cornell-box", 4, test_time=False, show_picture=True, show_result=True, _spp=1024)
    #test("cornell-box", 4, test_time=False, show_picture=True, show_result=True, _spp=1024, output_folder="result_light_hit_experiment")

    #test("cornell-box", 4, test_time=True, show_picture=False, show_result=True, _time=10, output_folder="epsilon_cornell_box_experiment")
    #update_total_result("../result_0414_compare_epsilon_opt/scale_2_time_40", test_time=True)
    #export_radiance("cornell-box", 4, n_uvs=[16], render_reference=False, export=True)

# total_dict["q_brdf_sarsa"] = {'sampling_strategy': SAMPLE_Q_COS_PROPORTION, 'q_table_update_method': Q_UPDATE_SARSA}
# total_dict["q_brdf_rej"] = {'sampling_strategy': SAMPLE_Q_COS_REJECT}
# total_dict["q_brdf_rej_sarsa"] = {'sampling_strategy': SAMPLE_Q_COS_REJECT, 'q_table_update_method': Q_UPDATE_SARSA}
# total_dict["q_brdf_mcmc"] = {'sampling_strategy': SAMPLE_Q_COS_MCMC}
# total_dict["q_brdf_mcmc_sarsa"] = {'sampling_strategy': SAMPLE_Q_COS_MCMC, 'q_table_update_method': Q_UPDATE_SARSA}
# total_dict["q"] = {'sampling_strategy': SAMPLE_Q_PROPORTION}
# total_dict["q_brdf"] = {'sampling_strategy': SAMPLE_Q_COS_PROPORTION}

# output_folder = '../test_20210311_n_cube_all'
# for k, v in total_dict.items():
# 	# Equal Sample
# 	for s in all_scenes:
# 		test2(s, scale=4, test_time=False, output_folder="%s/%s_equal_sample" % (output_folder, k), sample_type_name=k, **v)
# 	update_total_result("%s/%s_equal_sample" % (output_folder, k), False)
#
# 	# Equal Time
# 	for s in all_scenes:
# 		test2(s, scale=4, test_time=True, output_folder="%s/%s_equal_time" % (output_folder, k), sample_type_name=k, **v)
# 	update_total_result("%s/%s_equal_time" % (output_folder, k), True)

# test_multiple_and_export_result(all_scenes, 1, "../result_accum/scale_1_spp_1024", test_time=False)
# test_multiple_and_export_result(all_scenes, 1, "../result_accum/scale_1_time_60", test_time=True)
# test_multiple_and_export_result(all_scenes, 2, "../result_non_accum/scale_2_time_40", test_time=True)
# test_multiple_and_export_result(all_scenes, 2, "../result_non_accum/scale_2_spp_1024", test_time=False)
# all_scenes2 = ["veach_door_simple"]
# test_multiple_and_export_result(all_scenes, 4, "../result_accum_0309_1/scale_4_spp_1024", test_time=False)
# test_multiple_and_export_result(all_scenes2, 4, "../result_accum_0308_3/scale_4_time_5", test_time=True)

# make_reference_image_multiple(scene_names=all_scenes, scale=4)
# make_reference_image_multiple(scene_names=all_scenes, scale=2)
# make_reference_image_multiple(scene_names=all_scenes, scale=1)

# test_multiple_and_export_result(all_scenes, 4, "../result_accum/scale_4_time_5", test_time=True)

# update_total_result("../result/scale_4")
# all_scenes2 = ['living-room-2', 'living-room-3', 'staircase2', 'veach-ajar']
# all_scenes = ['cornell-box']
# make_reference_image(scale=8, scene_names=all_scenes)
# make_reference_image_multiple(scale=4, scene_names=all_scenes)
# make_reference_image_multiple(scale=2, scene_names=all_scenes)
# make_reference_image(scale=1, scene_names=all_scenes)

# make_reference_image_single("bathroom")

# test2("veach_door_simple", scale=4, show_result=True, show_picture=True, test_time=False, force_all_diffuse=False)
# test2("bathroom", scale=4, show_result=True, show_picture=True, test_time=False, force_all_diffuse=False)

# make_reference_image_single("cornell-box-hard", scale=2)
# test("cornell-box", scale=2)
# make_reference_image_single("staircase")
# test("bedroom")
