from path_guiding.directional_data_structure.directional_data_structure import DirectionalDataStructure
from pyoptix import Buffer
import numpy as np
from core.utils.math_utils import getDirectionFrom
from PIL import Image
import matplotlib.pyplot as plt


class DirectionalGrid(DirectionalDataStructure):
    def __init__(self, q_table, **kwargs):
        super().__init__()
        self.q_table = q_table
        self.n_uv = kwargs.get("n_uv", 16)
        self.directional_mapping_method = kwargs.get("directional_mapping_method", "cylindrical")

        if self.directional_mapping_method == "cylindrical":
            self.shape = (self.n_uv, self.n_uv)
        elif self.directional_mapping_method == "shirley":
            self.shape = (2 * self.n_uv, self.n_uv)
        self.vector_dots = np.array([0])

    def get_size(self):
        return int(np.prod(self.shape))

    def get_avg_size(self):
        return int(np.prod(self.shape))

    def create_optix_buffer(self, context):
        context['unitUVNumber'] = np.array(self.shape, dtype=np.uint32)
        context['unitUVNumber_inv'] = (1.0 / np.array(self.shape, dtype=np.float32)).astype(np.float32)
        N = int(np.prod(self.shape))
        input_array = np.zeros((N, 3), dtype=np.float32)
        for i in range(N):
            v = getDirectionFrom(i, (0.5, 0.5), (self.n_uv, self.n_uv), method=self.directional_mapping_method)
            input_array[i][0] = v[0]
            input_array[i][1] = v[1]
            input_array[i][2] = v[2]
        input_array_t = np.transpose(input_array)
        self.vector_dots = input_array @ input_array_t
        self.vector_dots = np.where(self.vector_dots > 0, self.vector_dots, 0)
        context['unitUVVectors'] = Buffer.from_array(input_array, dtype=np.float32, buffer_type='i', drop_last_dim=True)

        # if self.directional_mapping_method == "shirley":
        #     input_array = np.zeros((self.n_uv * self.n_uv * 2, 3), dtype=np.float32)
        #     for i in range(2 * self.n_uv * self.n_uv):
        #         v = getDirectionFrom(i, (0.5, 0.5), (self.n_uv, self.n_uv))
        #         input_array[i][0] = v[0]
        #         input_array[i][1] = v[1]
        #         input_array[i][2] = v[2]
        #     print(input_array)
        #     input_array_t = np.transpose(input_array)
        #     self.vector_dots = input_array @ input_array_t
        #     self.vector_dots = np.where(self.vector_dots > 0, self.vector_dots, 0)
        #     import matplotlib.pyplot as plt
        #     plt.imshow(self.vector_dots)
        #     plt.show()
        #     context['unitUVVectors'] = Buffer.from_array(input_array, dtype=np.float32, buffer_type='i', drop_last_dim=True)
        # elif self.directional_mapping_method == "cylindrical":
        #     input_array = np.zeros((self.n_uv * self.n_uv, 3), dtype=np.float32)
        #     for i in range(self.n_uv * self.n_uv):
        #         v = getDirectionFrom(i, (0.5, 0.5), (self.n_uv, self.n_uv), method=self.directional_mapping_method)
        #         input_array[i][0] = v[0]
        #         input_array[i][1] = v[1]
        #         input_array[i][2] = v[2]
        #     input_array_t = np.transpose(input_array)
        #     self.vector_dots = input_array @ input_array_t
        #     self.vector_dots = np.where(self.vector_dots > 0, self.vector_dots, 0)
        #     context['unitUVVectors'] = Buffer.from_array(input_array, dtype=np.float32, buffer_type='i', drop_last_dim=True)

    def __str__(self):
        logs = ["[Directional Data Structure]"]
        logs += ["\t- type : %s" % "Grid"]
        logs += ["\t- shape : %s" % str(self.shape)]
        logs += ["\t- directional mapping type : %s" % str(self.directional_mapping_method)]
        return "\n".join(logs)

    def visualize(self, index, image_size=512):
        value_array = self.q_table.q_table[index]
        value_array = value_array.reshape(self.shape)
        image = Image.fromarray(value_array)
        height = image_size
        width = int(image_size * self.shape[1] / self.shape[0])
        image = image.resize((width, height))
        image_np = np.asarray(image)
        plt.figure()
        plt.imshow(image_np)
        plt.show()
