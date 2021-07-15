from PIL import Image
import os
import glob
import numpy as np
from utils.image_utils import load_reference_image

crop_data = {
    "cornell-box-hard" : [(5,5, 20, 20)]
}


def tuple_to_str(t):
    return "_".join(map(str,t))


def crop_image_in_folder_randomly(folder, N):
    for f in glob.glob("%s/*/" % folder):
        scene_name = f.split("/")[-2]
        image_folder_name = "/".join([folder, scene_name, "images"])
        for f in glob.iglob("%s/*" % image_folder_name):
            image = Image.open(f)
            image_size = np.shape(image)
            break
        height, width, channel = image_size
        crop_data_list = []
        for i in range(N):
            while True:
                sx = np.random.randint(0, width)
                sy = np.random.randint(0, height)
                ex = sx + 64
                ey = sy + 64
                if ex < width and ey < height:
                    crop_data_list.append((sx, sy, ex, ey))
                    break

        for crop_data in crop_data_list:
            cropped_folder_output = "%s_cropped/%s/%s" % (folder, scene_name, tuple_to_str(crop_data))
            if not os.path.exists(cropped_folder_output):
                os.makedirs(cropped_folder_output)
            for f in glob.iglob("%s/*" % image_folder_name):
                file_name = f.split("/")[-1]
                image = Image.open(f)
                cropped_image = image.crop(crop_data)
                cropped_image = cropped_image.resize((256, 256))
                cropped_image.save(cropped_folder_output+"/"+file_name)
            # reference
            reference_parent_folder = '../../reference_images/%s/scale_%d' % ("standard", 2)
            target_name = "%s/%s_*.png" % (reference_parent_folder, scene_name)
            files = glob.glob(target_name)
            load_file = files[0]

            image = Image.open(load_file)
            cropped_image = image.crop(crop_data)
            cropped_image = cropped_image.resize((256, 256))
            cropped_image.save(cropped_folder_output + "/" + "reference.png")


def crop_image_in_folder(folder, crop_data):
    for scene_name, crop_data_list in crop_data.items():
        image_folder_name = "/".join([folder, scene_name, "images"])
        for crop_data in crop_data_list:
            cropped_folder_output = "%s_cropped/%s/%s" % (folder, scene_name, tuple_to_str(crop_data))
            if not os.path.exists(cropped_folder_output):
                os.makedirs(cropped_folder_output)
            for f in glob.iglob("%s/*" % image_folder_name):
                file_name = f.split("/")[-1]
                image = Image.open(f)
                cropped_image = image.crop(crop_data)
                cropped_image = cropped_image.resize((256, 256))
                cropped_image.save(cropped_folder_output+"/"+file_name)
            # reference
            reference_parent_folder = '../../reference_images/%s/scale_%d' % ("standard", 2)
            target_name = "%s/%s_*.png" % (reference_parent_folder, scene_name)
            files = glob.glob(target_name)
            load_file = files[0]

            image = Image.open(load_file)
            cropped_image = image.crop(crop_data)
            cropped_image = cropped_image.resize((256, 256))
            cropped_image.save(cropped_folder_output + "/" + "reference.png")


if __name__ == "__main__":
    #crop_image_in_folder_randomly("../../result_0412_2/scale_2_time_40", 6)
    crop_image_in_folder_randomly("../../result_0414_compare_epsilon_opt/scale_2_time_40", 12)

    #crop_image_in_folder("../../result_0412_2/scale_2_time_40", crop_data)