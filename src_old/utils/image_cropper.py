import os
import numpy as np
from PIL import Image


def crop_images_in(folder_name, sx, sy, ex, ey):
    listing = os.listdir(folder_name)
    print(listing)
    export_folder_name = folder_name+"_cropped_%d_%d_%d_%d" % (sx, sy, ex, ey)
    os.makedirs(export_folder_name, exist_ok=True)
    for file in listing:
        if not "png" in file:
            continue
        im = Image.open(folder_name +"/"+ file)
        area = (sx, sy, ex, ey)
        im = im.crop(area)
        im = im.resize((640, 360))
        im.save(export_folder_name +"/"+ file, "PNG")

# crop_images_in("../result/cornell_box_foggy_sigma_s_0.2000_hg_0.7000_16384",
#                394, 284, 494, 384)
# crop_images_in("../result/cornell_box_foggy_sigma_s_0.2000_hg_0.7000_16384",
#                14, 220, 114, 320)
# crop_images_in("../result/cornell_box_foggy_sigma_s_0.2000_hg_0.7000_16384",
#                26, 34, 126, 134)
# crop_images_in("../result/cornell_box_foggy_sigma_s_0.2000_hg_0.7000_16384",
#                213, 322, 314, 422)
# crop_images_in("../result/cornell_box_16384",
#                213, 322, 314, 422)
# crop_images_in("../result/veach_door_simple_16384",
#                131, 116, 131+160, 116+90)
# crop_images_in("../result/veach_door_simple_foggy_sigma_s_0.0010_hg_0.7000_16384",
#                750, 440, 750+160, 440+90)
# crop_images_in("../result/veach_door_simple_foggy_sigma_s_0.0010_hg_0.7000_16384",
#                402, 186, 402+160, 186+90)
# crop_images_in("../result/veach_door_simple_foggy_sigma_s_0.0030_hg_0.7000_16384",
#                750, 440, 750+160, 440+90)
# crop_images_in("../result/veach_door_simple_foggy_sigma_s_0.0030_hg_0.7000_16384",
#                402, 186, 402+160, 186+90)
