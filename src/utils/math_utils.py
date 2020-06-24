import numpy as np
import math


def normalize(mat):
    return mat / np.linalg.norm(mat)


def length2(vec):
    return vec.dot(vec)


class BoundingBox:
    def __init__(self, bbox_max=np.array([0, 0, 0], dtype=np.float32),
                 bbox_min=np.array([0, 0, 0], dtype=np.float32)):
        self.bbox_max = bbox_max
        self.bbox_min = bbox_min


def get_bbox_merged(bbox1: BoundingBox, bbox2: BoundingBox):
    # maxs = np.concatenate(())
    maxs = np.array([bbox1.bbox_max, bbox2.bbox_max])
    mins = np.array([bbox1.bbox_min, bbox2.bbox_min])
    new_max = np.amax(maxs, 0)
    new_min = np.amin(mins, 0)
    return BoundingBox(new_max, new_min)


def get_bbox_from_rectangle(o, u, v):
    v1 = o
    v2 = o + u
    v3 = o + v
    v4 = o + u + v
    vs = np.array([v1, v2, v3, v4])
    new_max = np.amax(vs, 0)
    new_min = np.amin(vs, 0)
    return BoundingBox(new_max, new_min)


def get_bbox_from_sphere(center, r):
    new_min = center - r
    new_max = center + r
    return BoundingBox(new_max, new_min)


def get_bbox_transformed(bbox: BoundingBox, transformation):
    tvs = []
    bbox_max = bbox.bbox_max
    bbox_min = bbox.bbox_min
    for i in range(8):
        bbox_x = bbox_max[0] if (i // 1) % 2 == 0 else bbox_min[0]
        bbox_y = bbox_max[1] if (i // 2) % 2 == 0 else bbox_min[1]
        bbox_z = bbox_max[2] if (i // 4) % 2 == 0 else bbox_min[2]

        v = np.array([bbox_x, bbox_y, bbox_z, 1])
        tv = transformation.dot(v)
        tv = v[0:3]
        tvs.append(tv)

    tvs = np.array(tvs, dtype=np.float32)
    print(tvs.shape)
    bbox_max_new = np.amax(tvs, 0)
    bbox_min_new = np.amin(tvs, 0)
    return BoundingBox(bbox_max_new, bbox_min_new)


def mapUVToDirection(uv, flipy=False):
    x = 2 * uv[0] - 1
    y = 2 * uv[1] - 1
    if (y > -x):
        if (y < x):
            xx = x
            if (y > 0):
                offset = 0
                yy = y
            else:
                offset = 7
                yy = x + y
        else:
            xx = y
            if (x > 0):
                offset = 1
                yy = y - x
            else:
                offset = 2
                yy = -x
    else:
        if (y > x):
            xx = -x
            if (y > 0):
                offset = 3
                yy = -x - y
            else:
                offset = 4
                yy = -y
        else:
            xx = -y
            if (x > 0):
                offset = 6
                yy = x
            else:
                if y != 0:
                    offset = 5
                    yy = x - y
                else:
                    return (0, 1, 0)
    assert xx >= 0
    theta = math.acos(1 - xx * xx)
    phi = (math.pi / 4) * (offset + (yy / xx))
    if flipy:
        ay = - math.cos(theta)
    else:
        ay = math.cos(theta)
    return (math.sin(theta) * math.cos(phi), ay, - math.sin(theta) * math.sin(phi))


def mapDirectionToUV(direction):
    M_PIf = math.pi
    Q_PIf = M_PIf / 4
    x = direction[0]
    y = -direction[2]
    theta = math.acos(abs(direction[1]))
    phi = math.atan2(y, x)
    phi += (2 * M_PIf)
    phi = phi % (2 * M_PIf)

    xx = math.sqrt(1 - math.cos(theta))
    offset = int(phi / Q_PIf)
    yy = phi / Q_PIf - float(offset)

    assert yy >= 0
    yy = yy * xx

    if (y > -x):
        if (y < x):
            u = xx
            if (y > 0):
                v = yy
            else:
                v = yy - u
        else:
            v = xx
            if (x > 0):
                u = v - yy
            else:
                u = -yy
    else:
        if (y > x):
            u = -xx
            if (y > 0):
                v = -u - yy
            else:
                v = -yy
        else:
            v = -xx
            if (x > 0):
                u = yy
            else:
                u = yy + v

    u = 0.5 * u + 0.5
    v = 0.5 * v + 0.5
    return (u, v)


def getDirectionFrom(index, offset, size):
    sx, sy = size
    u_index = (index // sy)
    v_index = (index % sy)
    inverted = False
    if u_index > sx:
        u_index -= sx
        inverted = True

    u_index_r = (float(u_index) + offset[0]) / (float(sx))
    v_index_r = (float(v_index) + offset[1]) / (float(sy))
    rx, ry, rz = mapUVToDirection((u_index_r, v_index_r))
    if inverted:
        return rx, -ry, rz
    else:
        return rx, ry, rz


def getEpsilon(index, max_index, t=0, a=0.1, k=100):
    if t is 0:
        return max(1 - index / max_index, 0)
    elif t is 1:
        x = math.pow(a, 1 / k)
        return math.pow(x, index)
