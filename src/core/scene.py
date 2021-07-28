from core.utils.math_utils import *
from pyoptix import GeometryInstance,  Transform, GeometryGroup, Acceleration
import os

from core.optix_mesh import OptixMesh
import xml.etree.ElementTree as ET
from utils.logging_utils import *
from utils.timing_utils import *
from utils.image_utils import *
from core.loader.loader_general import *
from core.shapes.objmesh import OBJMesh, InstancedShape, Shape
from itertools import chain
from core.textures.texture import *
from core.emitters.envmap import EnvironmentMap


def add_transform(transformation_matrix, geometry_instance):
    if transformation_matrix is None:
        transformation_matrix = np.eye(4, dtype=np.float32)
    else:
        transformation_matrix = np.array(transformation_matrix.transpose(), dtype=np.float32)

    gg = GeometryGroup(children=[geometry_instance])
    gg.set_acceleration(Acceleration("Trbvh"))

    transform = Transform(children=[gg])
    transform.set_matrix(False, transformation_matrix)
    transform.add_child(gg)
    return transform


class Scene:
    def __init__(self, name):
        self.name = name

        self.camera = None

        # name list (need to be virtually loaded after)
        self.texture_name_list = []

        self.texture_name_to_optix_index_dictionary = {}
        self.texture_sampler_list = []

        self.material_id_to_index_dictionary = {}
        self.material_list = []
        self.shape_list = []
        self.texture_list = []
        self.light_list = []

        self.obj_name_list = []
        self.obj_geometry_dict = {}

        self.geometry_instances = []
        self.light_instances = []

        self.folder_path = None
        self.bbox = BoundingBox()

        self.width = 0
        self.height = 0
        self.has_envmap = False

    @timing
    def load_scene_from(self, file_name):
        """
        Load scene from file.
        :param file_name: target file name
        :return:
        """
        doc = ET.parse(file_name)
        root = doc.getroot()

        scene_load_logger = load_logger('Scene config loader')
        shape_load_logger = load_logger('Shape config loader')
        material_load_logger = load_logger('Material config loader')
        emitter_load_logger = load_logger('Emitter config loader')

        sensor = root.find("sensor")

        # 0. load scene image size.
        film = load_film(sensor.find("film"))
        self.height, self.width = film.height, film.width
        # print log
        scene_load_logger.info("0. Image size loaded")
        scene_load_logger.info("[Size] : %dx%d" % (self.width, self.height))

        # 1. load camera
        self.camera = load_camera(root.find("sensor"))
        # print log
        scene_load_logger.info("1. Camera Loaded")
        scene_load_logger.info(str(self.camera))

        # 2. load geometry + material
        self.load_shapes(root)
        # print log
        shape_load_logger.info("2. Shape Loaded")
        shape_load_logger.info("Total %d shapes" % len(self.shape_list))
        for shape in self.shape_list:
            shape_load_logger.info(str(shape))
            shape_load_logger.info("\t- material id : %s" % shape.bsdf.id)
        material_load_logger.info("3. Material Loaded")
        material_load_logger.info("Total %d materials" % len(self.material_list))
        for material in self.material_list:
            material_load_logger.info(str(material))

        self.folder_path = os.path.dirname(file_name)

        # 3. load independent emitter info
        for emitter_node in root.findall('emitter'):
            emitter = load_emitter(emitter_node)
            if isinstance(emitter, EnvironmentMap):
                self.has_envmap = True
            emitter.list_index = len(self.light_list)
            self.light_list.append(emitter)

        # print log
        emitter_load_logger.info("4. Emitter Loaded")
        emitter_load_logger.info("Total %d emitters" % len(self.light_list))
        for light in self.light_list:
            emitter_load_logger.info(str(light))

    @timing
    def optix_create_geometry_instances(self, program_dictionary, material_dict, force_all_diffuse=False):
        Shape.program_dictionary = program_dictionary

        opaque_material = material_dict['opaque_material']
        cutout_material = material_dict['cutout_material']
        light_material = material_dict['light_material']

        geometry_instances = []
        light_instances = []

        for shape in self.shape_list:
            shape_type = shape.shape_type
            geometry = None
            bbox = None
            geometry = shape.to_optix_geometry()
            bbox = shape.get_bbox()

            # (1) create geometry
            if shape_type == "obj":
                mesh = self.obj_geometry_dict[shape.obj_file_name]
                shape.mesh = mesh
                geometry = mesh.geometry
                bbox = mesh.bbox
            elif shape_type == "rectangle":
                geometry = shape.to_optix_geometry()
                bbox = shape.get_bbox()
            elif shape_type == "sphere":
                geometry = shape.to_optix_geometry()
                bbox = shape.get_bbox()
            elif shape_type == "disk":
                geometry = shape.to_optix_geometry()
                bbox = shape.get_bbox()
            elif shape_type == "cube":
                geometry = shape.to_optix_geometry()
                bbox = shape.get_bbox()

            # (2) create material
            print(shape.bsdf.bsdf_type)
            if shape.emitter is not None:
                target_material = light_material
            else:
                bsdf_type = shape.bsdf.bsdf_type
                if bsdf_type == "mask":
                    target_material = cutout_material
                else:
                    target_material = opaque_material

            geometry_instance = GeometryInstance(geometry, target_material)
            mat_id = np.array(shape.bsdf.list_index, dtype=np.int32)
            emitter_id = np.array(shape.emitter.list_index if shape.emitter is not None else -1, dtype=np.int32)
            bsdf_type = np.array(int(shape.bsdf.optix_bsdf_type), dtype=np.int32)

            print("INFO", mat_id, emitter_id, bsdf_type)
            geometry_instance['materialId'] = mat_id
            geometry_instance["lightId"] = emitter_id
            geometry_instance['programId'] = bsdf_type

            # bsdf = shape.bsdf
            # geometry_instance = None
            # if material_parameter.type == "light":
            #     geometry_instance = GeometryInstance(geometry, light_material)
            #     geometry_instance["emission_color"] = material_parameter.emission
            #     geometry_instance["lightId"] = np.array(len(self.lights), dtype=np.int32)
            #     light = {"type": "area", "shape_data": shape, "emission": material_parameter.emission,
            #              "isTwosided": material_parameter.is_double_sided}
            #     self.lights.append(light)
            # else:
            #
            #     if material_parameter.is_cutoff:
            #         target_material = cutout_material
            #     else:
            #         target_material = opaque_material
                # if material_parameter.color0 is not None:
                #     geometry_instance = GeometryInstance(geometry, target_material)
                #     geometry_instance['programId'] = np.array(0, dtype=np.int32)
                #     geometry_instance['hasCheckerboard'] = np.array(1, dtype=np.int32)
                #     target_material['color0'] = material_parameter.color0
                #     target_material['color1'] = material_parameter.color1
                #     geometry_instance['to_uv'] = material_parameter.to_uv
                #     geometry_instance['materialId'] = np.array(material_parameter.uuid, dtype=np.int32)
                # el
                # if material_parameter.type == "diffuse":
                #     geometry_instance = GeometryInstance(geometry, target_material)
                #     geometry_instance['programId'] = np.array(0, dtype=np.int32)
                #     geometry_instance['materialId'] = np.array(material_parameter.uuid, dtype=np.int32)
                # elif material_parameter.type == "dielectric":
                #     geometry_instance = GeometryInstance(geometry, target_material)
                #     geometry_instance['programId'] = np.array(1, dtype=np.int32)
                #     geometry_instance['materialId'] = np.array(material_parameter.uuid, dtype=np.int32)
                # elif material_parameter.type == "roughdielectric":
                #     geometry_instance = GeometryInstance(geometry, target_material)
                #     geometry_instance['programId'] = np.array(2, dtype=np.int32)
                #     geometry_instance['materialId'] = np.array(material_parameter.uuid, dtype=np.int32)
                # elif material_parameter.type == "conductor":
                #     geometry_instance = GeometryInstance(geometry, target_material)
                #     geometry_instance['programId'] = np.array(3, dtype=np.int32)
                #     geometry_instance['eta'] = material_parameter.eta
                #     geometry_instance['k'] = material_parameter.k
                #     geometry_instance['materialId'] = np.array(material_parameter.uuid, dtype=np.int32)
                # elif material_parameter.type == "roughconductor":
                #     geometry_instance = GeometryInstance(geometry, target_material)
                #     geometry_instance['programId'] = np.array(4, dtype=np.int32)
                #     geometry_instance['eta'] = material_parameter.eta
                #     geometry_instance['k'] = material_parameter.k
                #     geometry_instance['materialId'] = np.array(material_parameter.uuid, dtype=np.int32)
                # elif material_parameter.type == "plastic":
                #     geometry_instance = GeometryInstance(geometry, target_material)
                #     geometry_instance['programId'] = np.array(5, dtype=np.int32)
                #     geometry_instance['materialId'] = np.array(material_parameter.uuid, dtype=np.int32)
                # elif material_parameter.type == "roughplastic":
                #     geometry_instance = GeometryInstance(geometry, target_material)
                #     geometry_instance['programId'] = np.array(6, dtype=np.int32)
                #     geometry_instance['materialId'] = np.array(material_parameter.uuid, dtype=np.int32)
                # elif material_parameter.type == "disney":
                #     geometry_instance = GeometryInstance(geometry, target_material)
                #     geometry_instance['programId'] = np.array(99, dtype=np.int32)
                #     geometry_instance['materialId'] = np.array(material_parameter.uuid, dtype=np.int32)
                # if force_all_diffuse:
                #     geometry_instance['programId'] = np.array(0, dtype=np.int32)

            if isinstance(shape, InstancedShape):
                bbox = get_bbox_transformed(bbox, np.array(shape.transform.transpose(), dtype=np.float32))

            # merge bbox
            self.bbox = get_bbox_merged(self.bbox, bbox)

            if isinstance(shape, OBJMesh):
                geometry_instance["faceNormals"] = np.array(1 if shape.face_normals else 0, dtype=np.int32)

            if isinstance(shape, InstancedShape):
                transform = add_transform(shape.transform, geometry_instance)
            else:
                transform = add_transform(None, geometry_instance)
            #if shape.transformation is not None:
            # geometry_instance["transformation"] = shape.transformation
            if target_material == light_material:
                light_instances.append(transform)
            else:
                geometry_instances.append(transform)

        # if self.name == "veach_door_simple":
        #     o = np.array([34.3580, 136.5705, -321.7834], dtype=np.float32)
        #     ox = np.array([-117.2283, 136.5705, -321.7834], dtype=np.float32)
        #     oy = np.array([34.3580, 76.5705, -321.7834], dtype=np.float32)
        #     u = ox - o
        #     v = oy - o
        #     geometry = create_parallelogram(o, u, v, par_int, par_bb)
        #     geometry_instance = GeometryInstance(geometry, light_material)
        #     emission_color = np.array([1420, 1552, 1642], dtype=np.float32)
        #     geometry_instance["emission_color"] = emission_color
        #     light_instances.append(add_transform(None, geometry_instance))
        #     shape = ShapeParameter()
        #     shape.shape_type = "rectangle"
        #     shape.rectangle_info = (o, u, v)
        #     light = {"type": "area", "shape_data": shape, "emission": emission_color}
        #     self.lights.append(light)

        self.geometry_instances = geometry_instances
        self.light_instances = light_instances

    @timing
    def optix_create_objs(self, program_dictionary):
        mesh_bb = program_dictionary['tri_mesh_bb']
        mesh_it = program_dictionary['tri_mesh_it']

        for obj_file_name in self.obj_name_list:
            mesh = OptixMesh(mesh_bb, mesh_it)
            mesh.load_from_file(self.folder_path + "/" + obj_file_name)
            self.obj_geometry_dict[obj_file_name] = mesh

    @timing
    def optix_load_textures(self):
        from core.textures.bitmap import BitmapTexture

        # environment map
        for light in self.light_list:
            if isinstance(light, EnvironmentMap):
                self.has_envmap = True
                tex_sampler = load_texture_sampler(self.folder_path, light.filename, gamma=1)
                self.texture_sampler_list.append(tex_sampler)
                light.envmapID = tex_sampler.get_id()
                print("ENV loaded", light.envmapID, light.filename)

        # get all materials
        self.texture_list = []
        for material in self.material_list:
            self.texture_list += material.get_textures()
        self.texture_name_list = []
        for texture in self.texture_list:
            if isinstance(texture, BitmapTexture) and texture.filename not in self.texture_name_list:
                self.texture_name_list.append(texture.filename)

        print("Load texture list")
        print(self.texture_name_list)

        for texture_name in self.texture_name_list:
            tex_sampler = load_texture_sampler(self.folder_path, texture_name, gamma=2.2)
            self.texture_name_to_optix_index_dictionary[texture_name] = tex_sampler.get_id()
            self.texture_sampler_list.append(tex_sampler)

        # assign optix id and list id to texture
        for (i, texture) in enumerate(self.texture_list):
            if isinstance(texture, BitmapTexture):
                texture.texture_optix_id = self.texture_name_to_optix_index_dictionary[texture.filename]
            texture.list_index = i

    def load_shapes(self, root):
        """
        Load shape information that includes material information, from root node
        :param root: root node
        :return:
        """
        shape_list = []
        obj_list = []
        anonymous_material_count = 0

        for node in root.findall('shape'):
            # 1. load shape
            shape = load_single_shape(node)

            # if shape includes obj mesh, this would be loaded after.
            if isinstance(shape, OBJMesh) and shape.obj_file_name not in obj_list:
                obj_list.append(shape.obj_file_name)

            # 2. load material
            material_ref = node.find("ref")

            # 2.1 defined by reference
            if material_ref is not None:
                bsdf_id = material_ref.attrib["id"]

                # 2.1.1 already loaded
                if bsdf_id in self.material_id_to_index_dictionary:
                    bsdf_index = self.material_id_to_index_dictionary[bsdf_id]
                    material = self.material_list[bsdf_index]

                # 2.2.2 not loaded
                else:
                    bsdf = root.find('bsdf[@id="%s"]' % bsdf_id)
                    material = self.load_new_material(bsdf)

            # 2.2 defined inside the node (anonymous material).
            else:
                bsdf_id = "anonymous_material_%d" % anonymous_material_count
                anonymous_material_count += 1
                bsdf = node.find("bsdf")
                material = self.load_new_material(bsdf, bsdf_id=bsdf_id)

            # 3. load emitter
            if node.find("emitter") is not None:
                from core.emitters.area import AreaLight
                emitter = load_emitter(node.find("emitter"))
                assert isinstance(emitter, AreaLight)
                shape.emitter = emitter
                emitter.shape = shape
                emitter.list_index = len(self.light_list)
                self.light_list.append(emitter)

                # print(shape.emitter)
                # radiance = load_value(emitter, "radiance", default=np.array([1, 1, 1], dtype=np.float32))
                # material_light = MaterialParameter("light_%d" % light_count)
                # light_count += 1
                # material_light.type = 'light'
                # material_light.is_double_sided = material.is_double_sided
                # material_light.emission = radiance
                # material = material_light

            shape.bsdf = material
            shape_list.append(shape)

        self.shape_list = shape_list
        self.obj_name_list = obj_list

    def load_new_material(self, bsdf, bsdf_id=None):
        """
        Load new material and save to material list
        :param bsdf: bsdf node
        :param bsdf_id: given bsdf name
        :return:
        """
        # load material information
        material = load_bsdf(bsdf)
        if material.id is None:
            material.id = bsdf_id

        # append material to material list
        self.material_id_to_index_dictionary[material.id] = len(self.material_list)
        material.list_index = len(self.material_list)
        self.material_list.append(material)

        return material
