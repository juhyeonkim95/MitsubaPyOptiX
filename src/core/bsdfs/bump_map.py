from core.bsdfs.bsdf import BSDF
from core.loader.loader_general import *


class BumpMap(BSDF):
    def __init__(self, props):
        super().__init__(props)
        self.texture = load_texture(props.find('texture'))
        self.bsdf = load_bsdf(props.find("bsdf"))
        self.optix_bsdf_type = self.bsdf.optix_bsdf_type

    def __str__(self):
        logs = ["[Material]"]
        logs += ["\t- type : %s" % "bump map"]
        logs += ["\t- texture : %s" % str(self.texture)]
        logs += ["\t- bsdf : %s" % str(self.bsdf)]
        return "\n".join(logs)

    def __array__(self):
        np_array = np.array(self.bsdf)
        # np_array["bumpmap_texture_id"] = self.texture.list_index
        return np_array
