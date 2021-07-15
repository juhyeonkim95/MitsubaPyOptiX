from PIL import Image, ImageDraw
import os
import glob
import numpy as np
from utils.image_utils import load_reference_image
SIZE = 256
SIZE_CROP = 64

_crop_data = {
    # Teaser
    # "kitchen": ["297_194_361_258", "274_281_338_345", "545_284_609_348"],

    #"bathroom": ["59_353_123_417", "342_37_406_101"],
    #"bathroom2": ["385_245_449_309", "125_152_189_216"],
    #"bedroom": ["417_238_481_302", "357_276_421_340"],
    #"cornell-box": ["92_432_156_496", "268_320_332_384"],
    #"cornell-box-hard": ["95_193_159_257", "274_342_338_406"],
    "kitchen": ["230_114_294_178", "469_271_533_335"],
    # "living-room": ["41_254_105_318", "462_206_526_270"],
    # "living-room-2": ["248_229_312_293", "524_168_588_232"],
    # "living-room-3": ["297_194_361_258", "274_281_338_345"],
    #"staircase": ["293_285_357_349", "19_196_83_260"],
    # "staircase2": ["403_421_467_485", "350_256_414_320"],
    #"veach_door_simple": ["344_44_408_108", "275_239_339_303"],
    #"veach-ajar": ["172_199_236_263", "220_247_284_311"]

    # "staircase2": ["248_229_312_293", "524_168_588_232"]
    # "kitchen": ["r"] * 10
    # "cornell-box": ["8_395_72_459", "268_320_332_384"],
}

target_colors = ["#FF0000", "#FFFF00", "#FFA500", "#FFFF00"] * 10

target_names = ["brdf",
                "q_mis_quadtree_mc",
                "q_brdf_rej_sarsa",
                "q_brdf_rej_expected_sarsa_mix",
                "q_brdf_rej_sarsa_mix",
                "reference"
                ]


def str_to_tuple(t):
    return tuple((map(int,t.split("_"))))


def tuple_to_str(t):
    return "_".join(map(str,t))


def crop_image_in_folder_randomly(folder, crop_data_lists):
    for scene_name, crop_data_list in crop_data_lists.items():
        image_folder_name = "/".join([folder, scene_name, "images"])
        cropped_image_stacked = np.zeros((SIZE * len(crop_data_list), SIZE * len(target_names), 3), dtype=np.uint8)

        cropped_folder_output = "%s_cropped/stacked" % (folder)
        if not os.path.exists(cropped_folder_output):
            os.makedirs(cropped_folder_output)

        reference_parent_folder = '../../reference_images/%s/scale_%d' % ("standard", 2)
        target_name = "%s/%s_*.png" % (reference_parent_folder, scene_name)
        files = glob.glob(target_name)
        file_name = files[0]
        reference_image = Image.open(file_name)
        width, height = reference_image.size
        print(width, height)
        crop_sizes = []
        for j, crop_data in enumerate(crop_data_list):
            if crop_data == "r":
                while True:
                    sx = np.random.randint(0, width)
                    sy = np.random.randint(0, height)
                    ex = sx + 64
                    ey = sy + 64
                    if ex < width and ey < height:
                        crop_size = (sx, sy, ex, ey)
                        break

            else:
                crop_size = str_to_tuple(crop_data)
            crop_sizes.append(crop_size)
        crop_sizes = sorted(crop_sizes, key=lambda tup: tup[0])

        for j, crop_size in enumerate(crop_sizes):
            print(tuple_to_str(crop_size))
            for i, name in enumerate(target_names):
                if name == "reference":
                    image = reference_image
                else:
                    file_name = "%s/%s.png" % (image_folder_name, name)
                    image = Image.open(file_name)

                cropped_image = image.crop(crop_size)
                cropped_image = cropped_image.resize((SIZE, SIZE))
                cropped_image_draw = ImageDraw.Draw(cropped_image)
                cropped_image_draw.rectangle((0, 0, SIZE, SIZE), width=5, outline=target_colors[j])

                sx = i * SIZE
                sy = j * SIZE
                cropped_image_stacked[sy:sy+SIZE, sx:sx+SIZE, :] = np.asarray(cropped_image)
            # reference

        #new_height = SIZE * 2
        #new_width = new_height / height * width

        reference_image_draw = ImageDraw.Draw(reference_image)
        for j, crop_size in enumerate(crop_sizes):
            reference_image_draw.rectangle(crop_size, width=5, outline=target_colors[j])

        # reference_image = reference_image.resize((new_width, new_height))

        cropped_image_stacked_image = Image.fromarray(cropped_image_stacked)
        cropped_image_stacked_image.save(cropped_folder_output + "/" + "%s_cropped_images.png" % scene_name)
        reference_image.save(cropped_folder_output + "/" + "%s_cropped_images_marked.png"% scene_name)


if __name__ == "__main__":
    #crop_image_in_folder_randomly("../../result_0412_2/scale_2_time_40", 6)
    crop_image_in_folder_randomly("../../result_0414_compare_epsilon_opt/scale_2_time_40", _crop_data)

    #crop_image_in_folder("../../result_0412_2/scale_2_time_40", crop_data)