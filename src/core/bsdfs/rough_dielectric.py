from core.bsdfs.bsdf import BSDF
from core.bsdfs.bsdf_flags import BSDFFlags
from core.loader.loader_general import *
from core.bsdfs.microfacet import *


class RoughDielectric(BSDF):
    def __init__(self, props):
        super().__init__(props)
        self.int_ior = load_value(props, "intIOR", default=1.5046)
        self.ext_ior = load_value(props, "extIOR", default=1.000277)
        self.specular_reflectance = load_value(props, "specularReflectance", default=1.0)
        self.specular_transmittance = load_value(props, "specularTransmittance", default=1.0)
        self.distribution = load_value(props, "distribution", "beckmann")
        self.alpha = load_value(props, "alpha", 0.1)
        self.optix_bsdf_type = BSDFFlags.rough_dielectric

    def __str__(self):
        logs = ["[Material]"]
        logs += ["\t- type : %s" % "rough_dielectric"]
        logs += ["\t- int_ior : %f" % self.int_ior]
        logs += ["\t- ext_ior : %f" % self.ext_ior]
        logs += ["\t- specular_reflectance : %s" % str(self.specular_reflectance)]
        logs += ["\t- specular_transmittance : %s" % str(self.specular_transmittance)]
        logs += ["\t- distribution : %s" % str(self.distribution)]
        logs += ["\t- alpha : %s" % str(self.alpha)]
        return "\n".join(logs)

    def __array__(self):
        from core.utils.array_utils import insert_texture_or_spectrum, insert_texture_or_spectrum_single_value
        np_array = np.zeros(1, dtype=BSDF.dtype)
        np_array['bsdf_type'] = self.optix_bsdf_type
        np_array['intIOR'] = float(self.int_ior)
        np_array['extIOR'] = float(self.ext_ior)
        np_array['distribution_type'] = int(DistributionType[self.distribution])
        insert_texture_or_spectrum(np_array, self.specular_reflectance, 'specular_reflectance')
        insert_texture_or_spectrum(np_array, self.specular_transmittance, 'specular_transmittance')
        insert_texture_or_spectrum_single_value(np_array, self.alpha, 'alpha')
        return np_array
