import numpy as np
from core.loader.loader_simple import *
from PIL import ImageColor


def load_spectrum(node, key="value", default=1):
    str_val = node.attrib[key]
    if str_val.startswith("#"):
        hex_to_rgb = ImageColor.getcolor(str_val, "RGB")
        rgb = np.array(list(hex_to_rgb), dtype=np.float32)
        rgb /= 255.0
        return rgb
    spectrum = str2floatarray(str_val)
    if len(spectrum) == 3:
        return spectrum
    elif len(spectrum) == 1:
        return np.repeat(spectrum, 3).astype(np.float32)
    else:
        raise NotImplementedError("Only rgb or single value is acceptable")