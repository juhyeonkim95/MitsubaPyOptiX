from core.bsdfs.bsdf import BSDF
from core.loader.loader_general import *


class TwoSided(BSDF):
    def __init__(self, props):
        super().__init__(props)
        self.bsdf = load_bsdf(props.find("bsdf"))
        self.optix_bsdf_type = self.bsdf.optix_bsdf_type

    def __str__(self):
        logs = ["[Material]"]
        logs += ["\t- type : %s" % "two sided"]
        logs += ["\t- bsdf : %s" % str(self.bsdf)]
        return "\n".join(logs)

    def __array__(self):
        np_array = np.array(self.bsdf)
        from core.bsdfs.dielectric import Dielectric
        from core.bsdfs.rough_dielectric import RoughDielectric

        # dielectric material cannot be two-sided.
        if not isinstance(self.bsdf, (Dielectric, RoughDielectric)):
            np_array["isTwosided"] = 1
        return np_array
