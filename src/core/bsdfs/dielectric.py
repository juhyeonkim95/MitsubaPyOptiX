from core.bsdfs.bsdf import BSDF
from core.bsdfs.bsdf_flags import BSDFFlags
from core.loader.loader_general import *


class Dielectric(BSDF):
    def __init__(self, props):
        super().__init__(props)
        self.int_ior = load_value(props, "intIOR", default=1.5046)
        self.ext_ior = load_value(props, "extIOR", default=1.000277)
        self.specular_reflectance = load_value(props, "specularReflectance", default=1.0)
        self.specular_transmittance = load_value(props, "specularTransmittance", default=1.0)
        self.optix_bsdf_type = BSDFFlags.dielectric

    def __str__(self):
        logs = ["[Material]"]
        logs += ["\t- type : %s" % "dielectric"]
        logs += ["\t- int_ior : %f" % self.int_ior]
        logs += ["\t- ext_ior : %f" % self.ext_ior]
        logs += ["\t- specular_reflectance : %s" % str(self.specular_reflectance)]
        logs += ["\t- specular_transmittance : %s" % str(self.specular_transmittance)]
        return "\n".join(logs)

    def __array__(self):
        from core.utils.array_utils import insert_texture_or_spectrum
        np_array = np.zeros(1, dtype=BSDF.dtype)
        np_array['bsdf_type'] = self.optix_bsdf_type
        np_array['intIOR'] = self.int_ior
        np_array['extIOR'] = self.ext_ior
        insert_texture_or_spectrum(np_array, self.specular_reflectance, 'specular_reflectance')
        insert_texture_or_spectrum(np_array, self.specular_transmittance, 'specular_transmittance')
        return np_array
