# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from test_utils import *
from pyoptix import Compiler


def make_reference_image_multiple(scene_names=None, scale=1, diffuse_only=False):
    renderer = Renderer(scale=scale)
    root_path = "../scene"
    if scene_names is None:
        scene_names = [os.path.relpath(f.path, root_path) for f in os.scandir(root_path) if f.is_dir()]
        scene_names.sort()
    print(scene_names)
    diffuse_folder = "diffuse_only" if diffuse_only else "standard"
    target_folder = '../reference_images_20210721/%s/scale_%d' % (diffuse_folder, scale)

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for scene_name in scene_names:
        common_params = {
            'scene_name': scene_name,
            '_spp': 2048 * 16,
            'samples_per_pass': 128,
            'max_depth': 16,
            'rr_begin_depth': 8,
        }
        try:
            image = renderer.render(**common_params, use_mis=False)
            file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_pred_images(image['image'], "%s/%s_%s" % (target_folder, scene_name, file_name))
        except Exception:
            print("Error")


if __name__ == '__main__':
    Compiler.clean()

    Compiler.keep_device_function = False
    file_dir = os.path.dirname(os.path.abspath(__file__))
    Compiler.add_program_directory(file_dir)

    from path_guiding.c_natives import update_binary_tree_native
    print(update_binary_tree_native)

    all_scenes1 = ['cornell-box', 'cornell-box-hard', "veach_door_simple"]
    all_scenes = all_scenes1 + ['bathroom', 'bathroom2', 'bedroom', 'kitchen',
                                'living-room'
                                'living-room-2', 'living-room-3', 'staircase', 'staircase2',
                                'veach-ajar']

    all_scenes = ['cornell-box']
    # all_scenes = ["material-testball"]

    # show_radiance_map("kitchen", 4)
    # test_single_scene("kitchen",
    #                   scale=4,
    #                   test_time=True,
    #                   show_picture=True,
    #                   show_result=True,
    #                   _spp=256,
    #                   _time=10,
    #                   test_target=1,
    #                   do_bsdf=True)
    # test_octree_build("cornell_box")
    # test_single_scene("cornell-box", test_target=6, visualize_octree=False, show_picture=True, show_result=True, do_bsdf=True)
    # test_single_scene("cornell-box", test_target=2, show_picture=True, show_result=True, do_bsdf=False, _spp=2048)
    #make_reference_image_multiple(all_scenes, scale=8)
    #make_reference_image_multiple(all_scenes, scale=4)
    #make_reference_image_multiple(all_scenes, scale=2)
    #make_reference_image_multiple(all_scenes, scale=1)

    for scene in all_scenes:
        test_single_scene(scene, scale=4, test_target=2, show_picture=True, show_result=True,
                          do_bsdf=False, _spp=1024, _time=10, test_time=True)

    #test_multiple_and_export_result(all_scenes, 4, output_folder="../result/sdtree_scale_4_time_10", _time=10, test_time=True, test_target=2)
    #test_multiple_and_export_result(all_scenes, 2, output_folder="../result_20210519_scale_2_time_40", _time=40, test_time=True)
    #test_multiple_and_export_result(all_scenes, 2, output_folder="../result_20210519_scale_2_time_20", _time=20, test_time=True)
    #test_multiple_and_export_result(all_scenes, 2, output_folder="../result_20210519_scale_2_time_10", _time=10, test_time=True)
