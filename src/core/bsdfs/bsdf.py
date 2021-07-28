from core.bsdfs.bsdf_flags import BSDFFlags
from core.textures.texture import Texture
from typing import List
import numpy as np


class BSDF:
    """
    Data type uploaded to optix
    Use single data structure that encompasses all types
    """
    dtype = np.dtype([
        ('bsdf_type', np.uint32),

        ('albedo', np.float32, 3),
        ('albedo_texture_id', np.int32),
        ('diffuse_reflectance', np.float32, 3),
        ('diffuse_reflectance_texture_id', np.int32),
        ('specular_reflectance', np.float32, 3),
        ('specular_reflectance_texture_id', np.int32),
        ('specular_transmittance', np.float32, 3),
        ('specular_transmittance_texture_id', np.int32),
        ('alpha', np.float32),
        ('alpha_texture_id', np.int32),
        ('opacity', np.float32),
        ('opacity_texture_id', np.int32),

        ('emission', np.float32, 3),
        ('metallic', np.float32),
        ('subsurface', np.float32),
        ('specular', np.float32),
        ('roughness', np.float32),
        ('specularTint', np.float32),
        ('anisotropic', np.float32),
        ('sheen', np.float32),
        ('sheenTint', np.float32),
        ('clearcoat', np.float32),
        ('clearcoatRoughness', np.float32),
        ('isTwosided', np.uint32),
        ('intIOR', np.float32),
        ('extIOR', np.float32),
        ('transmission', np.float32),
        ('ax', np.float32),
        ('ay', np.float32),
        ('distribution_type', np.int32),
        ('nonlinear', np.int32),
        ('bumpID', np.uint32),
        ('eta', np.float32, 3),
        ('k', np.float32, 3)
    ])

    def __init__(self, props):
        """
        BSDF class is only for upload data to OptiX
        :param props: property node
        """
        self.bsdf_type = props.attrib["type"]
        self.opacity = 1
        self.is_double_sided = True
        self.id = props.get("id", None)
        self.list_index = -1
        self.optix_bsdf_type = BSDFFlags.diffuse

    def __str__(self):
        pass

    def __array__(self):
        pass

    def get_textures(self) -> List[Texture]:
        """
        Get list of accompanying textures.
        :return: List of texture instances
        """
        textures = []
        members = [getattr(self, attr) for attr in dir(self) if not attr.startswith("__")]
        for member in members:
            if isinstance(member, Texture):
                textures.append(member)
            elif isinstance(member, BSDF):
                textures += member.get_textures()
        return textures
