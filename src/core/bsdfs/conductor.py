from core.bsdfs.bsdf import BSDF
from core.bsdfs.bsdf_flags import BSDFFlags
from core.loader.loader_general import *


class Conductor(BSDF):
    def __init__(self, props):
        super().__init__(props)
        self.eta = load_value(props, "eta", [0, 0, 0])
        self.k = load_value(props, "k", [1, 1, 1])
        self.specular_reflectance = load_value(props, "specularReflectance", 1.0)
        self.optix_bsdf_type = BSDFFlags.conductor

    def __str__(self):
        logs = ["[Material]"]
        logs += ["\t- type : %s" % "conductor"]
        logs += ["\t- eta : %s" % str(self.eta)]
        logs += ["\t- k : %s" % str(self.k)]
        logs += ["\t- specular_reflectance : %s" % str(self.specular_reflectance)]
        return "\n".join(logs)

    def __array__(self):
        from core.utils.array_utils import insert_texture_or_spectrum
        np_array = np.zeros(1, dtype=BSDF.dtype)
        np_array['bsdf_type'] = self.optix_bsdf_type
        np_array['k'] = np.array(self.k, dtype=np.float32)
        np_array['eta'] = np.array(self.eta, dtype=np.float32)
        insert_texture_or_spectrum(np_array, self.specular_reflectance, 'specular_reflectance')
        return np_array
