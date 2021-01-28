from pyoptix import Geometry
import numpy as np


def create_parallelogram(anchor, offset1, offset2, intersect_program, bb_program):
    parallelogram = Geometry(bounding_box_program=bb_program, intersection_program=intersect_program)
    parallelogram.set_primitive_count(1)
    normal = np.cross(offset1, offset2)
    normal /= np.linalg.norm(normal)

    d = np.dot(normal, anchor)
    plane = np.zeros(4, dtype=np.float32)
    plane[:3] = normal
    plane[3] = d

    v1 = offset1 / np.dot(offset1, offset1)
    v2 = offset2 / np.dot(offset2, offset2)

    parallelogram["plane"] = plane
    parallelogram["anchor"] = anchor
    parallelogram["v1"] = v1
    parallelogram["v2"] = v2
    return parallelogram


def create_sphere(center, radius, intersect_program, bb_program):
    sphere = Geometry(bounding_box_program=bb_program, intersection_program=intersect_program)
    sphere.set_primitive_count(1)
    sphere['sphere'] = np.array([center[0], center[1], center[2], radius], dtype=np.float32)
    return sphere


def create_disk(center, radius, normal, intersect_program, bb_program):
    disk = Geometry(bounding_box_program=bb_program, intersection_program=intersect_program)
    disk.set_primitive_count(1)
    disk["disk_pos_radii"] = np.array([center[0], center[1], center[2], radius], dtype=np.float32)
    disk["disk_normal"] = normal
    return disk
