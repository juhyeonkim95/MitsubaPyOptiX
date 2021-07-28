from core.utils.math_utils import *
from pyrr import Vector3
import math
from core.sensors.camera import Camera


class PerspectiveCamera(Camera):
    def __init__(self, props):
        """
        Perspective camera class.
        :param props : property node
        """
        from core.loader.loader_general import load_value
        super().__init__(props)
        self.fov = load_value(props, "fov", 35)
        self.fov_axis = load_value(props, "fovAxis", "x")
        transform = load_value(props, "toWorld")

        # camera position and vectors
        self.eye = np.array(transform * Vector3([0, 0, 0]))

        self.right = np.array(transform * Vector3([-1, 0, 0])) - self.eye
        self.up = np.array(transform * Vector3([0, 1, 0])) - self.eye
        self.forward = np.array(transform * Vector3([0, 0, 1])) - self.eye

        self.right = normalize(self.right)
        self.up = normalize(self.up)
        self.forward = normalize(self.forward)

    def calc_image_space_vectors(self, aspect_ratio):
        """
        Calculate image space vector from aspect ration
        :param aspect_ratio: width divide by height
        :return: u, v (right, up) in image space and w vector (camera's forward)
        """
        u = np.copy(self.right)
        v = np.copy(self.up)
        w = np.copy(self.forward)
        w_len = np.sqrt(np.sum(w ** 2))
        if self.fov_axis == "x":
            u_len = w_len * math.tan(0.5 * self.fov * math.pi / 180)
            v_len = u_len / aspect_ratio
        else:
            v_len = w_len * math.tan(0.5 * self.fov * math.pi / 180)
            u_len = v_len * aspect_ratio

        u *= u_len
        v *= v_len

        return u, v, w

    def __str__(self):
        logs = [
            "[Camera]",
            "\t- type : %s" % "perspective",
            "\t- position : %s" % str(self.eye),
            "\t- look at : %s" % str(self.eye + self.forward),
            "\t- right : %s" % str(self.right),
            "\t- up : %s" % str(self.up),
            "\t- forward : %s" % str(self.forward),
            "\t- fov (deg) : %s" % str(self.fov),
            "\t- fovAxis : %s" % str(self.fov_axis)
        ]
        return "\n".join(logs)
