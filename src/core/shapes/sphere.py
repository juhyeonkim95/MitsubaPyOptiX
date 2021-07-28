from core.shapes.shape import Shape
from pyrr import Vector3
from pyoptix import Geometry
import numpy as np
from core.utils.math_utils import BoundingBox
import math

class Sphere(Shape):
    def __init__(self, props):
        from core.loader.loader_general import load_value
        super().__init__(props)
        self.radius = load_value(props, "radius", 1.0)
        self.center = load_value(props, "center", Vector3([0, 0, 0]))

        if props.find("transform/matrix") is not None:
            transform = load_value(props, "toWorld")
            self.center = transform * self.center
            self.radius = np.linalg.norm(transform * Vector3([1, 0, 0]) - self.center)

        self.center = np.array(self.center, dtype=np.float32)
        self.radius = float(self.radius)

    def to_optix_geometry(self) -> Geometry:
        sphere = Geometry(
            bounding_box_program=Shape.program_dictionary["sphere_bb"],
            intersection_program=Shape.program_dictionary["sphere_it"]
        )
        sphere.set_primitive_count(1)
        sphere['sphere'] = np.append(self.center, self.radius).astype(np.float32)
        return sphere

    def get_bbox(self) -> BoundingBox:
        new_max = self.center + self.radius
        new_min = self.center - self.radius
        return BoundingBox(new_max, new_min)

    def __str__(self):
        logs = [
            "[Shape]",
            "\t- type : %s" % "sphere",
            "\t- center : %s" % str(self.center),
            "\t- radius : %s" % str(self.radius)
        ]
        return "\n".join(logs)

    def fill_area_light_array(self, np_array):
        np_array["lightType"] = 1
        np_array["position"] = np.array(self.center, dtype=np.float32)
        np_array["radius"] = self.radius
        np_array["area"] = 4 * math.pi * self.radius * self.radius
