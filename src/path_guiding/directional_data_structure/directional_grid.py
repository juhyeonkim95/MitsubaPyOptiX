from path_guiding.directional_data_structure.directional_data_structure import DirectionalDataStructure
from pyoptix import Buffer
import numpy as np
from core.utils.math_utils import getDirectionFrom


class DirectionalGrid(DirectionalDataStructure):
    def __init__(self, q_table, **kwargs):
        super().__init__()
        self.q_table = q_table
        self.n_uv = kwargs.get("n_uv", 16)

    def get_size(self):
        return 2 * self.n_uv * self.n_uv

    def create_optix_buffer(self, context):
        input_array = np.zeros((self.n_uv * self.n_uv * 2, 3), dtype=np.float32)
        for i in range(2 * self.n_uv * self.n_uv):
            v = getDirectionFrom(i, (0.5, 0.5), (self.n_uv, self.n_uv))
            input_array[i][0] = v[0]
            input_array[i][1] = v[1]
            input_array[i][2] = v[2]
        context['unitUVVectors'] = Buffer.from_array(input_array, dtype=np.float32, buffer_type='i', drop_last_dim=True)
        context['unitUVNumber'] = np.array([self.n_uv, self.n_uv], dtype=np.uint32)

    def __str__(self):
        logs = ["[Directional Data Structure]"]
        logs += ["\t- type : %s" % "Grid"]
        logs += ["\t- n_uv : %s" % str(self.n_uv)]
        return "\n".join(logs)