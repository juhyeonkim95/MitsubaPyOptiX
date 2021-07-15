import numpy as np
import math
from collections import OrderedDict


class Light:
    # Static data type
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
        ('indices_buffer_id', np.int32),
        ('normal_buffer_id', np.int32),
        ('n_triangles', np.int32),
        ('transformation', np.float32, (4, 4)),
        ('envmapID', np.int32),
        ('isTwosided', np.int32)
    ])

    def __init__(self, light_data):
        """
        Light data class.
        :param light_data: dictionary that contains light information
        """
        self.light_data = light_data
        self.np_array = self.create_np_array_from_data()

    def create_np_array_from_data(self):
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
                np_array["pos_buffer_id"] = shape_parameter.pos_buffer_id
                np_array["indices_buffer_id"] = shape_parameter.indice_buffer_id
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

    def __str__(self):
        infos = OrderedDict()
        infos["Type"] = self.light_data["type"]

        if self.light_data["type"] == "area":
            shape_parameter = self.light_data["shape_data"]
            shape_info = OrderedDict()
            infos["Shape Info"] = shape_info
            shape_info["Shape Type"] = shape_parameter.shape_type
            if shape_parameter.shape_type == "rectangle":
                shape_info["Position"] = self.np_array["position"]
                shape_info["Normal"] = self.np_array["normal"]
                shape_info["U"] = self.np_array["u"]
                shape_info["V"] = self.np_array["v"]
            elif shape_parameter.shape_type == "sphere":
                shape_info["Position"] = self.np_array["position"]
                shape_info["Radius"] = self.np_array["radius"]
            elif shape_parameter.shape_type == "disk":
                shape_info["Position"] = self.np_array["position"]
                shape_info["Radius"] = self.np_array["radius"]
                shape_info["Normal"] = self.np_array["normal"]
            infos["Emission"] = self.np_array["emission"]

        def recursive_print(d, level=0):
            s = []
            for k, v in d.items():
                if type(v) is OrderedDict:
                    s.append("\t" * level + "%s : " % k)
                    s = s + recursive_print(v, level + 1)
                else:
                    s.append("\t" * level + "%s : %s" % (k, str(v)))
            return s

        final_string = recursive_print(infos)
        final_string = "\n".join(["[Light]"] + final_string)

        return final_string


    """
    __array__ is called when a BasicLight is being converted to a numpy array.
    Then, one can assign that numpy array to an optix variable/buffer. The format will be user format.
    Memory layout (dtype) must match with the corresponding C struct in the device code.
    """
    def __array__(self):
        return self.np_array
