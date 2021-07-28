#from test_utils import *
#from pyoptix import Compiler
#import os
from test_utils_multiprocessing import *
from test_utils_multiprocessing_queued import *


if __name__ == '__main__':
    #render_using_multiprocessing("../configs/render_config_scale_4_time_10_2.json")

    render_using_multiprocessing("../configs/sdtree_comparison_20210727.json")

    #render_using_multiprocessing("../configs/render_config_scale_1_time_160.json")
    #render_using_multiprocessing("../configs/render_config_scale_1_spp_1024.json")
    #render_using_multiprocessing("../configs/render_config_scale_4_time_10.json")
    #render_using_multiprocessing("../configs/render_config_scale_2_time_40.json")
    #render_using_multiprocessing("../configs/render_config_scale_4_spp_1024.json")
    #render_using_multiprocessing("../configs/render_config_scale_2_spp_1024.json")
