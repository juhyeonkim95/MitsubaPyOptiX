from core.utils.math_utils import *


class Camera:
    def __init__(self, fov, fov_axis="x"):
        """
        Camera class. Camera is defined based on uvw vectors and eye position.
        Currently only supports perspective camera.
        :param fov: FOV (field of view) in degree
        :param fov_axis: FOV axis (default is x)
        """
        self.fov = fov
        self.fov_axis = fov_axis
        self.u = None
        self.v = None
        self.w = None
        self.eye = None

    def load_from_lookat(self, look_at, origin, up):
        """
        Load camera from look at parameters
        :param look_at: look at point
        :param origin: camera origin
        :param up: camera up vector
        :return:
        """
        self.w = normalize(look_at - origin)
        self.u = normalize(np.cross(self.w, up))
        self.v = normalize(np.cross(self.u, self.w))
        self.eye = origin

    def load_from_matrix(self, mat):
        """
        Load camera from camera (or view) matrix
        :param mat: camera matrix
        :return:
        """
        self.u = -normalize(mat[0:3, 0])
        self.v = normalize(mat[0:3, 1])
        self.w = normalize(mat[0:3, 2])
        self.eye = np.array(mat[0:3, 3])

    def __str__(self):
        logs = [
            "[Camera]",
            "position : %s" % str(self.eye),
            "look at : %s" % str(self.eye + self.w),
            "u : %s" % str(self.u),
            "v : %s" % str(self.v),
            "w : %s" % str(self.w)
        ]
        return "\n".join(logs)
