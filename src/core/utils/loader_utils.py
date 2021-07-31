import re
import numpy as np
from utils.image_utils import load_exr_image
from PIL import Image
from pyoptix import TextureSampler, Buffer
from core.utils.math_utils import srgb_to_linear
from utils.logging_utils import load_logger


def str_to_bool(s):
    if s == "true" or s == "True":
        return True
    elif s == "false" or s == "False":
        return False
    return False


def str2floatarray(s):
    s = re.sub(' +', ' ', s)
    s = re.sub(', ', ',', s)
    fs = re.split(",| ", s)
    float_array = [float(x) for x in fs]
    return np.array(float_array, dtype=np.float32)


def str2_4by4mat(s):
    ms = str2floatarray(s)
    ms = ms.reshape((4, 4))
    return ms


texture_load_logger = load_logger("Texture Loader")


def load_texture_sampler(folder_path, texture_name, gamma=-1):
    full_path = folder_path + "/" + texture_name
    if texture_name.endswith(".exr"):
        image = load_exr_image(full_path)
    elif texture_name.endswith(".pfm"):
        image = load_exr_image(full_path)
    elif texture_name.endswith(".hdr"):
        image = load_exr_image(full_path, True)
    else:
        image = Image.open(folder_path + "/" + texture_name).convert('RGBA')

    image_np = np.asarray(image)

    texture_load_logger.info("Name: %s , size: %s, dtype %s" % (full_path, str(image_np.shape), str(image_np.dtype)))
    # not linear color space --> need conversion to linear space
    if gamma != 1:
        def def_apply_gamma_rgb(image_rgb):
            image_rgb = image_rgb / 255.0
            image_rgb = srgb_to_linear(image_rgb, gamma)
            image_rgb = (image_rgb * 255.0).astype(np.uint8)
            return image_rgb
        image_np = np.array(image)
        image_np[:,:,0:3] = def_apply_gamma_rgb(image_np[:,:,0:3])

    tex_buffer = Buffer.from_array(image_np, buffer_type='i', drop_last_dim=True)

    tex_sampler = TextureSampler(tex_buffer,
                                 wrap_mode='repeat',
                                 indexing_mode='normalized_coordinates',
                                 read_mode='normalized_float',
                                 filter_mode='linear')
    return tex_sampler
