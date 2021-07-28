import numpy as np

#TEXTURE_BITMAP = 1
#TEXTURE_CHECKERBOARD = 2


class Texture:
    dtype = np.dtype([
        ('type', np.uint32),
        ('id', np.uint32),
        ('uv_transform', np.float32, (3, 3)),
        ('color0', np.float32, 3),
        ('color1', np.float32, 3),
        ('srgb', np.uint32)
    ])

    def __init__(self, props):
        """
        Texture class
        :param props: property node
        """
        self.list_index = -1
        self.type = props.attrib["type"]

    def __str__(self):
        pass

    def __array__(self):
        pass
