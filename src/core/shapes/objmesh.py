from core.shapes.shape import Shape, InstancedShape
from pyoptix import Geometry
import numpy as np
from core.utils.math_utils import BoundingBox


class OBJMesh(InstancedShape):
    def __init__(self, props):
        from core.loader.loader_general import load_value
        super().__init__(props)
        self.obj_file_name = load_value(props, "filename")
        self.face_normals = load_value(props, "faceNormals", default=False)
        self.flip_normals = load_value(props, "flipNormals", default=False)
        self.flip_tex_coords = load_value(props, "flipTexCoords", default=True)
        self.mesh = None

    def __str__(self):
        logs = [
            "[Shape]",
            "\t- type : %s" % "obj",
            "\t- filename : %s" % str(self.obj_file_name),
            "\t- face_normals : %s" % str(self.face_normals)
        ]
        return "\n".join(logs)

    def fill_area_light_array(self, np_array):
        np_array["lightType"] = 6
        np_array["pos_buffer_id"] = self.mesh.geometry['vertex_buffer'].get_id()
        np_array["indices_buffer_id"] = self.mesh.geometry['index_buffer'].get_id()
        np_array["normal_buffer_id"] = self.mesh.geometry['normal_buffer'].get_id()
        np_array["n_triangles"] = self.mesh.n_triangles
        np_array["transformation"] = np.array(self.transform.transpose(), dtype=np.float32)
        np_array["area"] = 1.0
