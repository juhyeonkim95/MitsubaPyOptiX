from core.shapes.shape import Shape, InstancedShape
from pyoptix import Geometry
import numpy as np
from core.utils.math_utils import BoundingBox


class Cube(InstancedShape):
    def __init__(self, props):
        super().__init__(props)

    def to_optix_geometry(self) -> Geometry:
        box = Geometry(
            bounding_box_program=Shape.program_dictionary["box_bb"],
            intersection_program=Shape.program_dictionary["box_it"]
        )
        box.set_primitive_count(1)
        box["boxmax"] = np.array([1, 1, 1], dtype=np.float32)
        box["boxmin"] = np.array([-1, -1, -1], dtype=np.float32)
        return box

    def get_bbox(self) -> BoundingBox:
        new_max = np.array([1, 1, 1], dtype=np.float32)
        new_min = np.array([-1, -1, -1], dtype=np.float32)
        return BoundingBox(new_max, new_min)

    def __str__(self):
        logs = [
            "[Shape]",
            "\t- type : %s" % "cube"
        ]
        return "\n".join(logs)
