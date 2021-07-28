from core.loader.loader_bsdf import load_bsdf
from core.loader.loader_camera import load_camera
from core.loader.loader_texture import load_texture
from core.loader.loader_shape import load_single_shape
from core.loader.loader_simple import *
from core.loader.loader_spectrum import load_spectrum
from core.loader.loader_emitter import load_emitter
from core.loader.loader_film import load_film
from core.utils.math_utils import *


def load_value(node, name, default=None, key="value"):
    """
    Load value from xml node by name
    :param node: xml node
    :param name: name of value
    :param default: default value
    :param key: key of load target. default is "value"
    :return: loaded value
    """
    child_node = node.find('*[@name="%s"]' % name)
    if child_node is not None:
        tag = child_node.tag
        # scalar / vector / matrix
        if tag == "integer":
            return load_int(child_node, key)
        if tag == "float":
            return load_float(child_node, key)
        elif tag == "boolean":
            return load_boolean(child_node, key)
        elif tag == "string":
            return load_string(child_node, key)
        elif tag == "point" or tag == "vector":
            return load_vector(child_node, default=0)
        elif tag == "transform":
            return load_transform(child_node)
        # Shape
        elif tag == "shape":
            return load_single_shape(child_node)
        # BSDF
        elif tag == "bsdf":
            return load_bsdf(child_node)
        # Emitter
        elif tag == "emitter":
            return load_emitter(child_node)
        # Sensor
        elif tag == "sensor":
            return load_camera(child_node)
        # Texture
        elif tag == "texture":
            return load_texture(child_node)
        # Spectra
        elif tag == "spectrum" or tag == "rgb" or tag == "srgb":
            value = load_spectrum(child_node)
            if tag == "srgb":
                value = srgb_to_linear(value)
            return value

        # film
        elif tag =="film":
            return load_film(node)
        else:
            raise NotImplementedError("current type %s is not implemented!" % tag)
    else:
        return default

