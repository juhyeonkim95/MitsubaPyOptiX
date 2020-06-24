import numpy as np
from pyoptix import Buffer, GeometryInstance, Geometry, Program, Material
from utils.math_utils import BoundingBox


class OptixMesh:
    mesh_bb = None
    mesh_it = None

    def __init__(self):
        pass
        self.geometry = Geometry(bounding_box_program=OptixMesh.mesh_bb, intersection_program=OptixMesh.mesh_it)
        #
        # # init buffers
        self.n_vertices = 0
        self.n_triangles = 0
        self.positions_buffer = Buffer.from_array([], dtype=np.dtype('f4, f4, f4'), buffer_type='i')
        self.tri_indices = Buffer.from_array([], dtype=np.dtype('i4, i4, i4'), buffer_type='i')
        self.normals_buffer = Buffer.from_array([], dtype=np.dtype('f4, f4, f4'), buffer_type='i')
        self.texcoord_buffer = Buffer.from_array([], dtype=np.dtype('f4, f4'), buffer_type='i')
        self.material_buffer = Buffer.from_array([], dtype=np.dtype('i4'), buffer_type='i')
        self.bbox = BoundingBox()

    def load_from_file(self, filename):
        vertices = []
        textures = []
        normals = []
        index_dictionary = {}
        vertices_reordered = []
        textures_reordered = []
        normals_reordered = []
        indices = []

        def add_face(triangle_info):
            tri_indices = []
            for v in triangle_info:
                w = v.split('/')
                w_tuple = tuple(w)
                if w_tuple not in index_dictionary:
                    index_dictionary[w_tuple] = len(index_dictionary)
                    vertices_reordered.append(vertices[int(w[0]) - 1])
                    textures_reordered.append(textures[int(w[1]) - 1])
                    normals_reordered.append(normals[int(w[2]) - 1])
                index = index_dictionary[w_tuple]
                tri_indices.append(index)
            return tri_indices

        for line in open(filename, 'r'):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue

            if values[0] == 'v':
                vertices.append(values[1:4])
            if values[0] == 'vt':
                textures.append(values[1:3])
            if values[0] == 'vn':
                normals.append(values[1:4])
            if values[0] == 'f':
                if len(values) == 5:
                    tri_indices = add_face(values[1:4])
                    indices.append(tri_indices)
                    tri_indices = add_face([values[3], values[4], values[1]])
                    indices.append(tri_indices)
                else:
                    tri_indices = add_face(values[1:4])
                    indices.append(tri_indices)

        vertices_np = np.asarray(vertices_reordered, dtype=np.float32)
        normals_np = np.asarray(normals_reordered, dtype=np.float32)
        textures_np = np.asarray(textures_reordered, dtype=np.float32)
        indices_np = np.asarray(indices, dtype=np.int32)

        self.bbox.bbox_max = np.amax(vertices_np, 0)
        self.bbox.bbox_max = np.amin(vertices_np, 0)

        self.n_triangles = indices_np.shape[0]
        self.n_vertices = vertices_np.shape[0]

        self.positions_buffer = Buffer.from_array(vertices_np, buffer_type='i', drop_last_dim=True)
        self.tri_indices = Buffer.from_array(indices_np, buffer_type='i', drop_last_dim=True)
        self.normals_buffer = Buffer.from_array(normals_np, buffer_type='i', drop_last_dim=True)
        self.texcoord_buffer = Buffer.from_array(textures_np, buffer_type='i', drop_last_dim=True)

        self.material_indices = np.zeros(self.n_triangles, np.int32)
        self.material_buffer = Buffer.from_array(self.material_indices, dtype=np.int32, buffer_type='i')

        self.geometry.set_primitive_count(self.n_triangles)
        self.geometry["vertex_buffer"] = self.positions_buffer
        self.geometry["index_buffer"] = self.tri_indices
        self.geometry["normal_buffer"] = self.normals_buffer
        self.geometry["texcoord_buffer"] = self.texcoord_buffer
        self.geometry["material_buffer"] = self.material_buffer