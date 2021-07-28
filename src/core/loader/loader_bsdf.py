from core.bsdfs.bsdf import BSDF


def load_bsdf(node) -> BSDF:
    """
    Loads bsdf from node
    :param node: bsdf node
    :return: BSDF instance
    """
    from core.bsdfs.diffuse import SmoothDiffuse
    from core.bsdfs.dielectric import Dielectric
    from core.bsdfs.rough_dielectric import RoughDielectric
    from core.bsdfs.conductor import Conductor
    from core.bsdfs.rough_conductor import RoughConductor
    from core.bsdfs.plastic import Plastic
    from core.bsdfs.rough_plastic import RoughPlastic
    from core.bsdfs.mask import Mask
    from core.bsdfs.bump_map import BumpMap
    from core.bsdfs.two_sided import TwoSided
    from core.bsdfs.coating import Coating

    bsdf_type = node.attrib['type']

    if bsdf_type == "diffuse":
        return SmoothDiffuse(node)
    elif bsdf_type == "dielectric" or bsdf_type == "thindielectric":
        return Dielectric(node)
    elif bsdf_type == "roughdielectric":
        return RoughDielectric(node)
    elif bsdf_type == "conductor":
        return Conductor(node)
    elif bsdf_type == "roughconductor":
        return RoughConductor(node)
    elif bsdf_type == "plastic":
        return Plastic(node)
    elif bsdf_type == "mask":
        return Mask(node)
    elif bsdf_type == "twosided":
        return TwoSided(node)
    elif bsdf_type == "bump" or bsdf_type == "bumpmap":
        return BumpMap(node)
    elif bsdf_type == "coating":
        return Coating(node)
    elif bsdf_type == "roughplastic":
        return RoughPlastic(node)
    else:
        raise NotImplementedError("BSDF type %s is not implemented!!" % bsdf_type)

