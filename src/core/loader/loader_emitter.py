from core.emitters.emitter import Emitter


def load_emitter(node) -> Emitter:
    """
    Load emitter
    :param node: sensor node
    :return: Camera object
    """
    from core.emitters.point import PointLight
    from core.emitters.area import AreaLight
    from core.emitters.spot import SpotEmitter
    from core.emitters.envmap import EnvironmentMap

    emitter_type = node.attrib['type']
    if emitter_type == "point":
        return PointLight(node)
    elif emitter_type == "area":
        return AreaLight(node)
    elif emitter_type == "spot":
        return SpotEmitter(node)
    elif emitter_type == "envmap":
        return EnvironmentMap(node)
    else:
        raise NotImplementedError("Emitter type %s is not implemented" % emitter_type)

