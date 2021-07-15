# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from test_utils import *
from pyoptix import Compiler


if __name__ == '__main__':
    Compiler.clean()

    Compiler.keep_device_function = False
    file_dir = os.path.dirname(os.path.abspath(__file__))
    Compiler.add_program_directory(file_dir)

    all_scenes1 = ['cornell-box', 'cornell-box-hard', "veach_door_simple"]
    all_scenes = all_scenes1 + ['bathroom', 'bathroom2', 'bedroom', 'kitchen',
                                'living-room', 'living-room-2', 'living-room-3', 'staircase', 'staircase2',
                                'veach-ajar']
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
    test_single_scene("cornell-box", test_target=2, show_picture=True, show_result=True, do_bsdf=False, _spp=2048)

    #test_multiple_and_export_result(all_scenes, 4, output_folder="../result_20210519_scale_4_time_10", _time=10, test_time=True)

    #test_multiple_and_export_result(all_scenes, 2, output_folder="../result_20210519_scale_2_time_40", _time=40, test_time=True)
    #test_multiple_and_export_result(all_scenes, 2, output_folder="../result_20210519_scale_2_time_20", _time=20, test_time=True)
    #test_multiple_and_export_result(all_scenes, 2, output_folder="../result_20210519_scale_2_time_10", _time=10, test_time=True)

