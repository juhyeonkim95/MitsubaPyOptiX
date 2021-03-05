from core.utils.math_utils import *


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
        self.u = normalize(np.cross(self.w, up))
        self.v = normalize(np.cross(self.u, self.w))
        self.eye = origin

    def load_from_matrix(self, mat):
        self.u = -normalize(mat[0:3, 0])
        self.v = normalize(mat[0:3, 1])
        self.w = normalize(mat[0:3, 2])
        self.eye = np.array(mat[0:3, 3])

    def print_lookat(self):
        print("pos:", self.eye, "look at:", self.eye+self.w)

    def print(self):
        print("U:", self.u, "V:", self.v, "W:", self.w, "Pos:", self.eye)