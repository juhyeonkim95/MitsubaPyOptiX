from core.shapes.shape import Shape


def load_single_shape(node) -> Shape:
    """
    Load single shape info from node
    :param node: shape node
    :return: shape
    """
    from core.shapes.rectangle import Rectangle
    from core.shapes.cube import Cube
    from core.shapes.sphere import Sphere
    from core.shapes.disk import Disk
    from core.shapes.objmesh import OBJMesh

    shape_type = node.attrib['type']
    if shape_type == "rectangle":
        return Rectangle(node)
    elif shape_type == "cube":
        return Cube(node)
    elif shape_type == "sphere":
        return Sphere(node)
    elif shape_type == "disk":
        return Disk(node)
    elif shape_type == "obj":
        return OBJMesh(node)
