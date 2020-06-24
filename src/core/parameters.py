import numpy as np
import hashlib


class MaterialParameter:
    def __init__(self, name):
        self.albedoID = -1
        seed = int(hashlib.sha256(name.encode('utf-8')).hexdigest(), 16) % 10000

        np.random.seed(seed)
        self.color = np.random.rand(3).astype(np.float32)
        self.emission = 0
        self.type = "diffuse"
        self.diffuse_map = -1
        self.intIOR = 1.0


class ShapeParameter:
    def __init__(self):
        self.shape_type = None
        self.rectangle_info = None
        self.radius = 0
        self.center = None
        self.obj_file_name = None
        self.transformation = None
        self.material_parameter = None