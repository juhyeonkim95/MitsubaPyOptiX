from utils.math_utils import *


class Camera:
    def __init__(self, fov, fov_axis="x"):
        self.fov = fov
        self.fov_axis = fov_axis
        self.u = None
        self.v = None
        self.w = None
        self.eye = None

    def load_from_lookat(self, look_at, origin, up):
        self.w = normalize(look_at - origin)
        self.v = normalize(up)
        self.u = normalize(np.cross(self.w, self.v))
        self.eye = origin

    def load_from_matrix(self, mat):
        self.u = normalize(mat[0:3, 0])
        self.v = normalize(mat[0:3, 1])
        self.w = normalize(mat[0:3, 2])
        self.eye = np.array(mat[0:3, 3])
