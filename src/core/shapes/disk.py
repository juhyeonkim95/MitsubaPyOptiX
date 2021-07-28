from core.shapes.shape import Shape
from pyrr import Vector3
from pyoptix import Geometry
import numpy as np
from core.utils.math_utils import BoundingBox
import math


class Disk(Shape):
    def __init__(self, props):
        super().__init__(props)
        from core.loader.loader_general import load_value
        transform = load_value(props, "toWorld")
        self.center = transform * Vector3([0, 0, 0], dtype=np.float32)
        self.radius = np.linalg.norm(transform * Vector3([1, 0, 0]) - self.center)
        self.normal = transform * Vector3([0, 0, 1], dtype=np.float32) - self.center

        self.center = np.array(self.center, dtype=np.float32)
        self.normal = np.array(self.normal, dtype=np.float32)
        self.radius = float(self.radius)

    def to_optix_geometry(self) -> Geometry:
        disk = Geometry(
            bounding_box_program=Shape.program_dictionary["disk_bb"],
            intersection_program=Shape.program_dictionary["disk_it"]
        )
        disk.set_primitive_count(1)
        disk["disk_pos_radii"] = np.append(self.center, self.radius).astype(np.float32)
        disk["disk_normal"] = self.normal
        return disk

    def get_bbox(self) -> BoundingBox:
        new_max = self.center + self.radius
        new_min = self.center - self.radius
        return BoundingBox(new_max, new_min)

    def __str__(self):
        logs = [
            "[Shape]",
            "\t- type : %s" % "disk",
            "\t- center : %s" % str(self.center),
            "\t- radius : %s" % str(self.radius),
            "\t- normal : %s" % str(self.normal)
        ]
        return "\n".join(logs)

    def fill_area_light_array(self, np_array):
        np_array["lightType"] = 5
        np_array["position"] = np.array(self.center, dtype=np.float32)
        np_array["normal"] = np.array(self.normal, dtype=np.float32)
        np_array["radius"] = self.radius
        np_array["area"] = math.pi * self.radius * self.radius
