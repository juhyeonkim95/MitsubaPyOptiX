import numpy as np
import hashlib
from enum import IntEnum

import math


class DistributionType(IntEnum):
    beckmann = 0
    phong = 1
    ggx = 2


class QuadTree(object):
    dtype = np.dtype([
        ('index_array', np.int32),
        ('rank_array', np.int32),
        ('depth_array', np.int32),

        ('color', np.float32, 3),
        ('casts_shadow', np.int32),
        ('padding', np.int32),
    ])

    """
    __array__ is called when a BasicLight is being converted to a numpy array.
    Then, one can assign that numpy array to an optix variable/buffer. The format will be user format.
    Memory layout (dtype) must match with the corresponding C struct in the device code.
    """
    def __array__(self):
        np_array = np.zeros(1, dtype=QuadTree.dtype)
        np_array['pos'] = self._pos
        np_array['color'] = self._color
        np_array['casts_shadow'] = 1 if self._casts_shadow else 0
        np_array['padding'] = 0
        return np_array

    def __init__(self, pos, color, casts_shadow):
        self._pos = pos
        self._color = color
        self._casts_shadow = casts_shadow


class MaterialParameter:
    dtype = np.dtype([
        ('albedoID', np.int32),
        ('albedo', np.float32, 3),
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
        ('opacity', np.float32),
        ('isTwosided', np.uint32),
        ('intIOR', np.float32),
        ('extIOR', np.float32),
        ('transmission', np.float32),
        ('ax', np.float32),
        ('ay', np.float32),
        ('distribution_type', np.int32),
        ('nonlinear', np.int32)
    ])

    def __init__(self, name):
        self.name = name

        seed = int(hashlib.sha256(name.encode('utf-8')).hexdigest(), 16) % 10000
        np.random.seed(seed)

        self.diffuse_map_id = 0

        self.color = np.random.rand(3).astype(np.float32)
        self.random_color = np.copy(self.color)
        self.emission = 0
        self.metallic = 0.0
        self.subsurface = 0.0
        self.specular = 0.5
        self.roughness = 0.5
        self.specularTint = 0.0
        self.anisotropic = 0.0
        self.sheen = 0.0
        self.sheenTint = 0.0
        self.clearcoat = 0.0
        self.clearcoatRoughness = 0.001
        self.opacity = 1.0
        self.is_double_sided = False
        self.intIOR = 1.45
        self.extIOR = 1.0
        self.transmission = 0

        self.type = "diffuse"
        self.diffuse_map = None
        self.is_cutoff = False
        self.uuid = 0

        self.color0 = None
        self.color1 = None
        self.to_uv = None

        self.eta = 0
        self.k = 0

        self.nonlinear = False
        self.distribution_type = "ggx"

    def print(self):
        print(self.type, self.name, self.color, self.is_cutoff, self.opacity)

    def __array__(self):
        np_array = np.zeros(1, dtype=MaterialParameter.dtype)
        np_array['albedoID'] = self.diffuse_map_id
        np_array['albedo'] = self.color
        np_array['emission'] = self.emission
        np_array['metallic'] = self.metallic
        np_array['subsurface'] = self.subsurface
        ior = self.intIOR / self.extIOR
        t = (ior - 1) / (ior + 1)
        self.specular = t * t / 0.08
        np_array['specular'] = self.specular
        np_array['roughness'] = self.roughness
        np_array['specularTint'] = self.specularTint
        np_array['anisotropic'] = self.anisotropic
        np_array['sheen'] = self.sheen
        np_array['sheenTint'] = self.sheenTint
        np_array['clearcoat'] = self.clearcoat
        np_array['clearcoatRoughness'] = self.clearcoatRoughness
        np_array['opacity'] = self.opacity
        np_array['isTwosided'] = np.array(1 if self.is_double_sided else 0, dtype=np.uint32)
        np_array['intIOR'] = self.intIOR
        np_array['extIOR'] = self.extIOR
        np_array['transmission'] = self.transmission
        np_array['distribution_type'] = int(DistributionType[self.distribution_type])
        np_array['nonlinear'] = 1 if self.nonlinear else 0

        aspect = math.sqrt(1.0 - self.anisotropic * 0.9)
        np_array['ax'] = max(0.001, self.roughness / aspect)
        np_array['ay'] = max(0.001, self.roughness * aspect)

        return np_array


class ShapeParameter:
    def __init__(self):
        self.shape_type = None
        self.rectangle_info = None
        self.triangle_info = None
        self.radius = 0
        self.center = None
        self.normal = None
        self.obj_file_name = None
        self.transformation = None
        self.material_parameter = None
        self.pos_buffer_id = 0
        self.indice_buffer_id = 0
        self.normal_buffer_id = 0
        self.n_triangles = 0
        self.face_normals = False


class EmitterParameter:
    def __init__(self):
        self.emitter_type = None
        self.intensity = None
        self.radiance = None
