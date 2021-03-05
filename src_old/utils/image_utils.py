import numpy as np
import cv2 as cv
from PIL import Image
import glob
import matplotlib.pyplot as plt
import os


def load_exr_image(path, invert=False):
    image = cv.imread(path, -1).astype(np.float32)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGRA)
    if invert:
        image = cv.flip(image, 0)
    return image


def convert_image_to_uint(image):
    x = np.copy(image)
    x *= 255
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return x


def save_pred_images(images, file_path):
    x = convert_image_to_uint(images)
    new_im = Image.fromarray(x)
    dirname = os.path.dirname(file_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    new_im.save("%s.png" % file_path)


def load_reference_image(parent_folder, name):
    target_name = "%s/%s_*.png" % (parent_folder, name)
    files = glob.glob(target_name)
    load_file = files[0]
    return load_image(load_file)


def load_image(path):
    image = Image.open(path)
    image = np.asarray(image, dtype=np.float32)
    image = image[:, :, 0:3]
    image /= 255.0
    return image


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
