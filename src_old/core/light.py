# import numpy as np
#
#
# class Light:
#     dtype = np.dtype([
#         ('position', np.float32, 3),
#         ('normal', np.float32, 3),
#         ('emission', np.float32, 3),
#         ('u', np.float32, 3),
#         ('v', np.float32, 3),
#         ('radius', np.float32),
#         ('lightType', np.uint32)
#     ])
#
#     def __init__(self):
#         # structure : [corner, v1, v2, normal, emission] (all float3)
#         self.buffer_numpy = np.empty(1, dtype=Light.dtype)
#
#     @property
#     def buffer(self):
#         return self.buffer_numpy.tobytes()
#
#
#     """
#     __array__ is called when a BasicLight is being converted to a numpy array.
#     Then, one can assign that numpy array to an optix variable/buffer. The format will be user format.
#     Memory layout (dtype) must match with the corresponding C struct in the device code.
#     """
#     # def __array__(self):
#     #     np_array = np.zeros(1, dtype=BasicLight.dtype)
#     #     np_array['pos'] = self._pos
#     #     np_array['color'] = self._color
#     #     np_array['casts_shadow'] = 1 if self._casts_shadow else 0
#     #     np_array['padding'] = 0
#     #     return np_array
#     #
#     def set_info(self, light_data):
#         shape_parameter = light_data["shape_data"]
#         if shape_parameter.shape_type == "rectangle":
#             o, u, v = shape_parameter.rectangle_info
#             normal = np.cross(u, v)
#             normal /= np.linalg.norm(normal)
#             self.buffer_numpy["position"] = o
#             self.buffer_numpy["normal"] = normal
#             self.buffer_numpy["u"] = u
#             self.buffer_numpy["v"] = v
#             self.buffer_numpy["lightType"] = 0
#             self.buffer_numpy["radius"] = 0.0
#         elif shape_parameter.shape_type == "sphere":
#             self.buffer_numpy["position"] = shape_parameter.center
#             self.buffer_numpy["radius"] = shape_parameter.radius
#             self.buffer_numpy["lightType"] = 1
#         self.buffer_numpy["emission"] = light_data["emission"]
import numpy as np
import math


class Light:
    dtype = np.dtype([
        ('position', np.float32, 3),
        ('direction', np.float32, 3),
        ('normal', np.float32, 3),
        ('emission', np.float32, 3),
        ('intensity', np.float32, 3),
        ('u', np.float32, 3),
        ('v', np.float32, 3),
        ('radius', np.float32),
        ('area', np.float32),
        ('cosTotalWidth', np.float32),
        ('cosFalloffStart', np.float32),
        ('lightType', np.uint32),
        ('pos_buffer_id', np.int32),
        ('indice_buffer_id', np.int32),
        ('normal_buffer_id', np.int32),
        ('n_triangles', np.int32),
        ('transformation', np.float32, (4, 4)),
        ('envmapID', np.int32),
        ('isTwosided', np.int32)
    ])

    def __init__(self, light_data):
        # structure : [corner, v1, v2, normal, emission] (all float3)
        # self.buffer_numpy = np.empty((1, ), dtype=Light.dtype)
        self.light_data = light_data

    # @property
    # def buffer(self):
    #     return self.buffer_numpy.tobytes()
    #

    """
    __array__ is called when a BasicLight is being converted to a numpy array.
    Then, one can assign that numpy array to an optix variable/buffer. The format will be user format.
    Memory layout (dtype) must match with the corresponding C struct in the device code.
    """
    def __array__(self):
        np_array = np.zeros(1, dtype=Light.dtype)

        if self.light_data["type"] == "area":
            shape_parameter = self.light_data["shape_data"]
            print(shape_parameter.shape_type, "Shape type")
            if shape_parameter.shape_type == "rectangle":
                o, u, v = shape_parameter.rectangle_info
                normal = np.cross(u, v)
                normal /= np.linalg.norm(normal)
                np_array["position"] = o
                np_array["normal"] = normal
                np_array["u"] = u
                np_array["v"] = v
                np_array["lightType"] = 0
                np_array["radius"] = 0.0
                np_array["area"] = np.linalg.norm(np.cross(u, v))
            elif shape_parameter.shape_type == "sphere":
                np_array["position"] = shape_parameter.center
                np_array["radius"] = shape_parameter.radius
                np_array["lightType"] = 1
                np_array["area"] = 4 * math.pi * shape_parameter.radius * shape_parameter.radius
            elif shape_parameter.shape_type == "disk":
                np_array["position"] = shape_parameter.center
                np_array["radius"] = shape_parameter.radius
                np_array["normal"] = shape_parameter.normal
                np_array["lightType"] = 5
                np_array["area"] = math.pi * shape_parameter.radius * shape_parameter.radius
            elif shape_parameter.shape_type == "obj":
                print("OBJ type light added")
                print("Position buffer id :", shape_parameter.pos_buffer_id)
                print("Indice buffer id :", shape_parameter.indice_buffer_id)
                print("Normal buffer id :", shape_parameter.normal_buffer_id)
                print("Face number :", shape_parameter.n_triangles)
                print("Transformation :", shape_parameter.transformation)
                np_array["pos_buffer_id"] = shape_parameter.pos_buffer_id
                np_array["indice_buffer_id"] = shape_parameter.indice_buffer_id
                np_array["normal_buffer_id"] = shape_parameter.normal_buffer_id
                np_array["n_triangles"] = shape_parameter.n_triangles
                np_array['transformation'] = shape_parameter.transformation
                np_array["lightType"] = 6
            np_array["emission"] = self.light_data["emission"]

        elif self.light_data["type"] == "point":
            np_array["lightType"] = 2
            np_array["intensity"] = self.light_data["intensity"]
            np_array["position"] = self.light_data["position"]
        elif self.light_data["type"] == "directional":
            np_array["lightType"] = 3
            np_array["emission"] = self.light_data["emission"]
            np_array["direction"] = self.light_data["direction"]
        elif self.light_data["type"] == "spot":
            np_array["lightType"] = 4
            np_array["intensity"] = self.light_data["intensity"]
            np_array["position"] = self.light_data["position"]
            np_array["direction"] = self.light_data["direction"]
            np_array["cosTotalWidth"] = math.cos(math.radians(self.light_data["cutoffAngle"]))
            np_array["cosFalloffStart"] = math.cos(math.radians(self.light_data["beamWidth"]))
        elif self.light_data["type"] == "envmap":
            np_array["lightType"] = 5
            np_array["envmapID"] = self.light_data['envmapID']
            np_array["transformation"] = self.light_data['transformation']
        np_array["isTwosided"] = 1 if self.light_data.get('isTwosided', False) else 0

        return np_array
