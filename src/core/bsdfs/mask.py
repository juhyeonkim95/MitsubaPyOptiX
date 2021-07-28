from core.bsdfs.bsdf import BSDF
from core.loader.loader_general import *


class Mask(BSDF):
    def __init__(self, props):
        super().__init__(props)
        self.opacity = load_value(props, "opacity", 0.5)
        self.bsdf = load_bsdf(props.find("bsdf"))
        self.optix_bsdf_type = self.bsdf.optix_bsdf_type

    def __str__(self):
        logs = ["[Material]"]
        logs += ["\t- type : %s" % "mask"]
        logs += ["\t- opacity : %s" % str(self.opacity)]
        logs += ["\t- bsdf : %s" % str(self.bsdf)]
        return "\n".join(logs)

    def __array__(self):
        from core.utils.array_utils import insert_texture_or_spectrum_single_value
        np_array = np.array(self.bsdf)
        insert_texture_or_spectrum_single_value(np_array, self.opacity, 'opacity')
        return np_array
