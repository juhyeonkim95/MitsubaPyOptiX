from core.bsdfs.bsdf import BSDF
from core.bsdfs.bsdf_flags import BSDFFlags
from core.loader.loader_general import *


class RoughPlastic(BSDF):
    def __init__(self, props):
        super().__init__(props)
        self.int_ior = load_value(props, "intIOR", default=1.5046)
        self.ext_ior = load_value(props, "extIOR", default=1.000277)
        self.diffuse_reflectance = load_value(props, "diffuseReflectance", default=[0.5, 0.5, 0.5])
        self.specular_reflectance = load_value(props, "specularReflectance", default=[1, 1, 1])
        self.nonlinear = load_value(props, "nonlinear", default=False)
        self.optix_bsdf_type = BSDFFlags.diffuse

    def __str__(self):
        logs = ["[Material]"]
        logs += ["\t- type : %s" % "rough_plastic"]
        logs += ["\t- int_ior : %f" % self.int_ior]
        logs += ["\t- ext_ior : %f" % self.ext_ior]
        logs += ["\t- diffuse_reflectance : %s" % str(self.diffuse_reflectance)]
        logs += ["\t- specular_reflectance : %s" % str(self.specular_reflectance)]
        logs += ["\t- nonlinear : %s" % str(self.nonlinear)]
        return "\n".join(logs)

    def __array__(self):
        from core.utils.array_utils import insert_texture_or_spectrum
        np_array = np.zeros(1, dtype=BSDF.dtype)
        np_array['bsdf_type'] = self.optix_bsdf_type
        insert_texture_or_spectrum(np_array, self.diffuse_reflectance, 'diffuse_reflectance')
        insert_texture_or_spectrum(np_array, self.specular_reflectance, 'specular_reflectance')
        return np_array
