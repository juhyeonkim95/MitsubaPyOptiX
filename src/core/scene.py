from utils.scene_loader import *
from core.camera import *
from utils.math_utils import *
from utils.geometry_creator import *
from pyoptix import Program, Material, GeometryInstance, \
    TextureSampler, Buffer, Transform, GeometryGroup, Group, Acceleration
import os
from PIL import Image
from core.optix_mesh import OptixMesh


def add_transform(transformation_matrix, geometry_instance):
    if transformation_matrix is None:
        transformation_matrix = np.eye(4, dtype=np.float32)

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
        self.texture_name_list = []
        self.texture_name_id_dictionary = {}
        self.texture_sampler_list = []

        self.material_dictionary = {}
        self.shape_list = []
        self.obj_list = []
        self.obj_geometry_dict = {}

        self.geometry_instances = []
        self.light_instances = []

        self.folder_path = None
        self.bbox = BoundingBox()
        self.lights = []
        self.width = 0
        self.height = 0

    def load_scene_from(self, file_name):
        doc = ET.parse(file_name)
        root = doc.getroot()

        # 0. load scene image size.
        self.height, self.width = load_scene_size(root.find("sensor"))
        print("Size: %d x %d" % (self.width, self.height))

        # 1. load camera
        self.camera = load_camera(root.find("sensor"))

        # 2. load material info
        self.texture_name_list, self.material_dictionary = load_material(root)

        # 3. load shape info
        self.shape_list, self.obj_list = load_shape(root, self.material_dictionary)

        # 4. load emitter info
        self.lights = load_emitter(root)

        print("Material dictionary", self.material_dictionary.keys())

        self.folder_path = os.path.dirname(file_name)

    def create_object_instances(self):
        par_bb = Program('optix/shapes/parallelogram.cu', 'bounds')
        par_int = Program('optix/shapes/parallelogram.cu', 'intersect')
        sph_bb = Program('optix/shapes/sphere.cu', 'bounds')
        sph_int = Program('optix/shapes/sphere.cu', 'intersect')
        disk_bb = Program('optix/shapes/disk.cu', 'bounds')
        disk_int = Program('optix/shapes/disk.cu', 'intersect')

        # target_program = 'optix/integrators/optixPathTracerQTable.cu'
        # a = Transform()
        # a.add_child()

        # diffuse = Material(closest_hit={0: Program(diffuse_target_cu, 'diffuse')} ,any_hit={1: Program(diffuse_target_cu, 'shadow')})
        target_program = 'optix/integrators/light_hit_program.cu'
        diffuse_light = Material(closest_hit={0: Program(target_program, 'diffuseEmitter')},
                                 any_hit={1: Program(target_program, 'any_hit')})
        diffuse_target_cu = 'optix/integrators/hit_program.cu'
        diffuse = Material(closest_hit={0: Program(diffuse_target_cu, 'closest_hit')},
                           any_hit={1: Program(diffuse_target_cu, 'any_hit')})

        diffuse["programId"] = np.array(0, dtype=np.int32)
        # glass = Material(closest_hit={0: Program(target_program, 'glass')})
        # glass["refraction_index"] = np.array([1.4], dtype=np.float32)
        # glass["refraction_color"] = np.array([0.99, 0.99, 0.99], dtype=np.float32)
        # glass["reflection_color"] = np.array([0.99, 0.99, 0.99], dtype=np.float32)
        # glass["extinction"] = np.array([0, 0, 0], dtype=np.float32)

        geometry_instances = []
        light_instances = []

        for shape in self.shape_list:
            shape_type = shape.shape_type
            geometry = None
            bbox = None
            if shape_type == "obj":
                mesh = self.obj_geometry_dict[shape.obj_file_name]
                shape.pos_buffer_id = mesh.geometry['vertex_buffer'].get_id()
                shape.indice_buffer_id = mesh.geometry['index_buffer'].get_id()
                shape.normal_buffer_id = mesh.geometry['normal_buffer'].get_id()
                shape.n_triangles = mesh.n_triangles
                geometry = mesh.geometry
                bbox = mesh.bbox
            elif shape_type == "rectangle":
                o, u, v = shape.rectangle_info
                geometry = create_parallelogram(o, u, v, par_int, par_bb)
                bbox = get_bbox_from_rectangle(o, u, v)
            elif shape_type == "sphere":
                bbox = get_bbox_from_sphere(shape.center, shape.radius)
                geometry = create_sphere(shape.center, shape.radius, sph_int, sph_bb)
            elif shape_type == "disk":
                bbox = get_bbox_from_sphere(shape.center, shape.radius)
                geometry = create_disk(shape.center, shape.radius, shape.normal, disk_int, disk_bb)

            material_parameter = shape.material_parameter
            geometry_instance = None
            if material_parameter.type == "light":
                geometry_instance = GeometryInstance(geometry, diffuse_light)
                geometry_instance["emission_color"] = material_parameter.emission
                geometry_instance["lightId"] = np.array(len(self.lights), dtype=np.int32)
                light = {"type": "area", "shape_data": shape, "emission": material_parameter.emission}
                self.lights.append(light)

            elif material_parameter.type == "diffuse":
                print("diffuse", material_parameter.color)
                if material_parameter.diffuse_map != -1:
                    geometry_instance = GeometryInstance(geometry, diffuse)
                    material_parameter.albedoID = self.texture_name_id_dictionary[material_parameter.diffuse_map]
                    geometry_instance["diffuse_map_id"] = np.array(material_parameter.albedoID, dtype=np.int32)
                else:
                    geometry_instance = GeometryInstance(geometry, diffuse)
                    geometry_instance["diffuse_color"] = material_parameter.color
                    geometry_instance["diffuse_map_id"] = np.array(0, dtype=np.int32)
            elif material_parameter.type == "dielectric":
                print("dielectric!!")
                geometry_instance = GeometryInstance(geometry, diffuse)
                geometry_instance["diffuse_color"] = material_parameter.color
                geometry_instance["diffuse_map_id"] = np.array(0, dtype=np.int32)
                #geometry_instance = GeometryInstance(geometry, diffuse)
                #geometry_instance["refraction_index"] = np.array([material_parameter.intIOR], dtype=np.float32)

            if shape.transformation is not None:
                bbox = get_bbox_transformed(bbox, shape.transformation)
                self.bbox = get_bbox_merged(self.bbox, bbox)

            transform = add_transform(shape.transformation, geometry_instance)
            #if shape.transformation is not None:
            # geometry_instance["transformation"] = shape.transformation
            if material_parameter.type == "light":
                light_instances.append(transform)
            else:
                geometry_instances.append(transform)
        if self.name == "veach_door_simple":
            o = np.array([34.3580, 136.5705, -321.7834], dtype=np.float32)
            ox = np.array([-117.2283, 136.5705, -321.7834], dtype=np.float32)
            oy = np.array([34.3580, 76.5705, -321.7834], dtype=np.float32)
            u = ox - o
            v = oy - o
            geometry = create_parallelogram(o, u, v, par_int, par_bb)
            geometry_instance = GeometryInstance(geometry, diffuse_light)
            emission_color = np.array([1420, 1552, 1642], dtype=np.float32)
            geometry_instance["emission_color"] = emission_color
            light_instances.append(add_transform(None, geometry_instance))
            shape = ShapeParameter()
            shape.shape_type = "rectangle"
            shape.rectangle_info = (o, u, v)
            light = {"type": "area", "shape_data": shape, "emission": emission_color}
            self.lights.append(light)

        self.geometry_instances = geometry_instances
        self.light_instances = light_instances
        print(self.bbox.bbox_max)
        print(self.bbox.bbox_min)

    def create_objs(self):
        OptixMesh.mesh_bb = Program('optix/shapes/triangle_mesh.cu', 'mesh_bounds')
        OptixMesh.mesh_it = Program('optix/shapes/triangle_mesh.cu', 'mesh_intersect')

        for obj_file_name in self.obj_list:
            mesh = OptixMesh()
            mesh.load_from_file(self.folder_path + "/" + obj_file_name)
            self.obj_geometry_dict[obj_file_name] = mesh

    def load_images(self):
        print(self.texture_name_list)
        i = 1
        for texture_name in self.texture_name_list:
            image = Image.open(self.folder_path + "/" + texture_name).convert('RGBA')
            image_np = np.asarray(image)#.astype(np.float32)
            #image_np = image_np / 255
            #print(image_np.dtype)
            #print(image_np.shape)
            tex_buffer = Buffer.from_array(image_np, buffer_type='i', drop_last_dim=True)

            tex_sampler = TextureSampler(tex_buffer,
                                         wrap_mode='repeat',
                                         indexing_mode='normalized_coordinates',
                                         read_mode='normalized_float',
                                         filter_mode='linear')

            self.texture_name_id_dictionary[texture_name] = tex_sampler.get_id()
            print(texture_name, tex_sampler.get_buffer().get_size())
            print(texture_name, i)
            print(texture_name, tex_sampler.get_buffer().dtype)
            self.texture_sampler_list.append(tex_sampler)
            i += 1



    def create_materials(self):
        for material_parameter in self.material_dictionary:
            pass


        # camera_fov, camera_matrix, self.material_dictionary, \
        # self.shape_dictionary = load_scene(file_name)
        # self.camera.load_from_matrix(camera_matrix)
        # self.camera.fov = camera_fov

