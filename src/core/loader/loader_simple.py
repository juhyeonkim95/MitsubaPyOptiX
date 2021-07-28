from pyrr import Vector3, Matrix44, Quaternion
from core.utils.loader_utils import *
from core.utils.math_utils import normalize


def load_int(node, key, default=0):
    return int(node.attrib.get(key, default))


def load_float(node, key, default=0):
    return float(node.attrib.get(key, default))


def load_boolean(node, key, default=False):
    if key in node.attrib:
        return str_to_bool(node.attrib[key])
    else:
        return default


def load_string(node, key, default=""):
    return node.attrib.get(key, default)


def load_vector(node, key=None, default=0) -> Vector3:
    """
    Load vector from node
    :param node:
    :param key: key name
    :param default: default value
    :return: Vector3
    """
    # used for "x=, y=, z=" format
    # ex) vector, point, rotate axis, translate, scale
    if key is None:
        x = load_float(node, "x", default)
        y = load_float(node, "y", default)
        z = load_float(node, "z", default)
        return Vector3([x, y, z], dtype=np.float32)
    # used for key="1, 2, 3" format
    # ex) "origin="2, 3, 4"
    else:
        return Vector3(str2floatarray(node.attrib[key]))


def load_transform(transform_node) -> Matrix44:
    """
    Load 4x4 transformation matrix from transform node
    :param transform_node: node that contains transformation information
    :return: 4x4 numpy matrix
    """
    transform_matrix = Matrix44.identity()

    all_transforms = transform_node.findall('*')
    for node in all_transforms:
        current_transform = Matrix44.identity()
        tag = node.tag
        if tag == "matrix":
            current_transform = Matrix44(str2_4by4mat(node.attrib['value']).transpose())
        elif tag == "translate":
            translation = load_vector(node)
            current_transform = Matrix44.from_translation(translation)
        elif tag == "scale":
            if node.attrib["value"] is not None:
                scale = load_float(node, "value")
            else:
                scale = load_vector(node, key=None, default=1)
            current_transform = Matrix44.from_scale(scale)
        elif tag == "rotate":
            axis = load_vector(node)
            angle = load_float(node, "angle")
            current_transform = Quaternion.from_axis_rotation(axis, angle)
        elif tag == 'lookat':
            origin = np.asarray(load_vector(node, 'origin'))
            target = np.asarray(load_vector(node, 'target'))
            up = np.asarray(load_vector(node, 'up'))

            forward = normalize(target - origin)
            right = normalize(np.cross(forward, up))
            up = normalize(np.cross(right, forward))

            matrix = np.array([
                [-right[0], -right[1], -right[2], 0],
                [up[0], up[1], up[2], 0],
                [forward[0], forward[1], forward[2], 0],
                [origin[0], origin[1], origin[2], 1]
            ])

            current_transform = Matrix44(matrix)

        transform_matrix = transform_matrix * current_transform

    return transform_matrix
