from core.sensors.camera import Camera


def load_camera(node) -> Camera:
    """
    Load camera info from sensor node. Only perspective camera.
    :param node: sensor node
    :return: Camera object
    """
    from core.sensors.perspective import PerspectiveCamera

    camera_type = node.attrib['type']
    if camera_type == "perspective":
        return PerspectiveCamera(node)
