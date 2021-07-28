from core.textures.texture import Texture


def load_texture(node) -> Texture:
    from core.textures.bitmap import BitmapTexture
    from core.textures.checkerboard import CheckerBoard
    from core.textures.scale import ScalingTexture

    texture_type = node.attrib['type']
    if texture_type == "bitmap":
        return BitmapTexture(node)
    elif texture_type == "checkerboard":
        return CheckerBoard(node)
    elif texture_type == "scale":
        return ScalingTexture(node)
    else:
        raise NotImplementedError("Texture type %s is not implemented!!" % texture_type)
