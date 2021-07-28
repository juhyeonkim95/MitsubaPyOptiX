from core.bsdfs.bsdf import BSDF
from core.bsdfs.bsdf_flags import BSDFFlags
from core.loader.loader_general import *


class SmoothDiffuse(BSDF):
    def __init__(self, props):
        super().__init__(props)
        self.reflectance = load_value(props, "reflectance", 0.5)
        self.optix_bsdf_type = BSDFFlags.diffuse

    def __str__(self):
        logs = ["[Material]"]
        logs += ["\t- type : %s" % "diffuse"]
        logs += ["\t- reflectance : %s" % str(self.reflectance)]
        return "\n".join(logs)

    def __array__(self):
        from core.utils.array_utils import insert_texture_or_spectrum
        np_array = np.zeros(1, dtype=BSDF.dtype)
        np_array['bsdf_type'] = self.optix_bsdf_type
        insert_texture_or_spectrum(np_array, self.reflectance, 'diffuse_reflectance')
        return np_array
