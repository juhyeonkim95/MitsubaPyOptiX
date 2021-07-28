from pyoptix import Geometry
import numpy as np
from core.utils.math_utils import *
from pyrr import Vector3, matrix44


class Shape:
    """
    Shape information class
    """
    program_dictionary = {}

    def __init__(self, props):
        self.shape_type = props.attrib["type"]
        self.bsdf = None
        self.emitter = None

    def to_optix_geometry(self) -> Geometry:
        pass

    def get_bbox(self) -> BoundingBox:
        pass

    def __str__(self):
        pass

    def fill_area_light_array(self, np_array):
        pass


class InstancedShape(Shape):
    """
    Instanced Shape information class.
    They are defined by original shape and transform.
    """
    def __init__(self, props):
        from core.loader.loader_general import load_value
        super().__init__(props)
        self.transform = load_value(props, "toWorld")

    def to_optix_geometry(self) -> Geometry:
        pass

    def get_bbox(self) -> BoundingBox:
        pass



