from path_guiding.spatial_data_structure.spatial_data_structure import SpatialDataStructure
import numpy as np


class SpatialVoxel(SpatialDataStructure):
    def __init__(self, q_table, **kwargs):
        super().__init__()
        self.q_table = q_table
        self.n_cube = kwargs.get("n_cube", 16)

    def __str__(self):
        logs = ["[Spatial Data Structure]"]
        logs += ["\t- type : %s" % "Voxel"]
        logs += ["\t- shape : %s" % str(self.n_cube)]
        return "\n".join(logs)

    def get_max_size(self):
        return self.n_cube * self.n_cube * self.n_cube

    def get_size(self):
        return self.n_cube * self.n_cube * self.n_cube

    def create_optix_buffer(self, context):
        context['unitCubeNumber'] = np.array([self.n_cube] * 3, dtype=np.uint32)
