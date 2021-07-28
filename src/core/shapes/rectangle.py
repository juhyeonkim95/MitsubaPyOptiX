from core.shapes.shape import Shape
from pyrr import Vector3
from pyoptix import Geometry
import numpy as np
from core.utils.math_utils import BoundingBox


class Rectangle(Shape):
    def __init__(self, props):
        from core.loader.loader_general import load_value
        super().__init__(props)
        transform = load_value(props, "toWorld")
        self.anchor = transform * Vector3([-1, -1, 0], dtype=np.float32)
        self.offset1 = transform * Vector3([1, -1, 0], dtype=np.float32) - self.anchor
        self.offset2 = transform * Vector3([-1, 1, 0], dtype=np.float32) - self.anchor

        self.anchor = np.array(self.anchor, dtype=np.float32)
        self.offset1 = np.array(self.offset1, dtype=np.float32)
        self.offset2 = np.array(self.offset2, dtype=np.float32)
        normal = np.cross(self.offset1, self.offset2)
        normal /= np.linalg.norm(normal)
        self.normal = normal

    def to_optix_geometry(self) -> Geometry:
        parallelogram = Geometry(
            bounding_box_program=Shape.program_dictionary["quad_bb"],
            intersection_program=Shape.program_dictionary["quad_it"]
        )
        parallelogram.set_primitive_count(1)
        d = np.dot(self.normal, self.anchor)
        plane = np.zeros(4, dtype=np.float32)
        plane[:3] = self.normal
        plane[3] = d

        v1 = self.offset1 / np.dot(self.offset1, self.offset1)
        v2 = self.offset2 / np.dot(self.offset2, self.offset2)

        parallelogram["plane"] = plane
        parallelogram["anchor"] = self.anchor
        parallelogram["v1"] = v1
        parallelogram["v2"] = v2
        return parallelogram

    def get_bbox(self) -> BoundingBox:
        v1 = self.anchor
        v2 = self.anchor + self.offset1
        v3 = self.anchor + self.offset2
        v4 = self.anchor + self.offset1 + self.offset2
        vs = np.array([v1, v2, v3, v4])
        new_max = np.amax(vs, 0)
        new_min = np.amin(vs, 0)
        return BoundingBox(new_max, new_min)

    def __str__(self):
        logs = [
            "[Shape]",
            "\t- type : %s" % "rectangle",
            "\t- anchor : %s" % str(self.anchor),
            "\t- offset1 : %s" % str(self.offset1),
            "\t- offset2 : %s" % str(self.offset2)
        ]
        return "\n".join(logs)

    def fill_area_light_array(self, np_array):
        np_array["lightType"] = 0
        np_array["position"] = np.array(self.anchor, dtype=np.float32)
        np_array["normal"] = np.array(self.normal, dtype=np.float32)
        u = np.array(self.offset1, dtype=np.float32)
        v = np.array(self.offset2, dtype=np.float32)
        np_array["u"] = u
        np_array["v"] = v
        np_array["area"] = np.linalg.norm(np.cross(u, v))
