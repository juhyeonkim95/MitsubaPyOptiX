from core.loader.loader_bsdf import load_bsdf
from core.loader.loader_camera import load_camera
from core.loader.loader_texture import load_texture
from core.loader.loader_shape import load_single_shape


def load_value(node, name, default=None, key="value"):
    """
    Load value from xml node
    :param node: xml node
    :param name: name of value
    :param default: default value
    :param key:
    :return: loaded value
    """
    child_node = node.find('*[@name="%s"]' % name)
    if child_node is not None:
        value = child_node.attrib.get(key, default)
        tag = child_node.tag
        if tag == "float":
            return float(value)
        elif tag == "integer":
            return int(value)
        elif tag == "boolean":
            return str_to_bool(value)
        elif tag == "string":
            return value
        elif tag == "transform":
            return load_transform(child_node)
        elif tag == "point":
            return load_vector(child_node, default=0)
        elif tag == "spectrum" or tag == "rgb":
            return load_spectrum(child_node)
        elif tag == "texture":
            return load_texture(child_node)
        return value
    else:
        return default