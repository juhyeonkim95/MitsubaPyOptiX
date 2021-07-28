from core.bsdfs.bsdf import BSDF
from core.loader.loader_general import *


class Coating(BSDF):
    def __init__(self, props):
        super().__init__(props)
        self.int_ior = load_value(props, "intIOR", default=1.5046)
        self.ext_ior = load_value(props, "extIOR", default=1.000277)
        self.thickness = load_value(props, "thickness", default=1)
        self.sigmaA = load_value(props, "sigmaA", default=0)
        self.specular_reflectance = load_value(props, "specularReflectance", default=[1, 1, 1])
        self.bsdf = load_bsdf(props.find("bsdf"))
        self.optix_bsdf_type = self.bsdf.optix_bsdf_type

    def __str__(self):
        logs = ["[Material]"]
        logs += ["\t- type : %s" % "coating"]
        logs += ["\t- bsdf : %s" % str(self.bsdf)]
        return "\n".join(logs)

    def __array__(self):
        np_array = np.array(self.bsdf)
        return np_array
