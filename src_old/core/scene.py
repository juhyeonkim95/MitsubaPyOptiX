from core.scene_loader import *
from core.utils.math_utils import *
from core.utils.geometry_creator import *
from pyoptix import GeometryInstance, \
    TextureSampler, Buffer, Transform, GeometryGroup, Acceleration
import os
from PIL import Image
from core.optix_mesh import OptixMesh
import xml.etree.ElementTree as ET


def add_transform(transformation_matrix, geometry_instance):
    if transformation_matrix is None:
        transformation_matrix = np.eye(4, dtype=np.float32)
    print(transformation_matrix)
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

        self.texture_name_id_dictionary = {}
        self.texture_sampler_list = []

        self.material_name_to_id_dictionary = {}
        self.material_list = []
        self.shape_list = []

        self.obj_name_list = []
        self.obj_geometry_dict = {}

        self.geometry_instances = []
        self.light_instances = []

        self.folder_path = None
        self.bbox = BoundingBox()
        self.lights = []
        self.width = 0
        self.height = 0
        self.annonymous_material_count = 0
        self.has_envmap = False

    def load_scene_from(self, file_name):
        doc = ET.parse(file_name)
        root = doc.getroot()

        # 0. load scene image size.
        self.height, self.width = load_scene_size(root.find("sensor"))
        print("Size: %d x %d" % (self.width, self.height))

        # 1. load camera
        self.camera = load_camera(root.find("sensor"))

        # 2. load geometry + material
        self.load_shapes(root)

        self.folder_path = os.path.dirname(file_name)

        # 3. load independent emitter info
        self.lights, env_map_file = load_emitter(root, self.folder_path)

        if env_map_file is not None:
            self.has_envmap = True
            self.texture_name_list.append(env_map_file)

        print("Mat list _____________________")
        for m in self.material_list:
            m.print()

    def optix_create_geometry_instances(self, program_dictionary, material_dict, force_all_diffuse=False):
        par_bb = program_dictionary["quad_bb"]
        par_int = program_dictionary["quad_it"]
        sph_bb = program_dictionary["sphere_bb"]
        sph_int = program_dictionary["sphere_it"]
        disk_bb = program_dictionary["disk_bb"]
        disk_int = program_dictionary["disk_it"]

        opaque_material = material_dict['opaque_material']
        cutout_material = material_dict['cutout_material']
        light_material = material_dict['light_material']

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
                geometry_instance = GeometryInstance(geometry, light_material)
                geometry_instance["emission_color"] = material_parameter.emission
                geometry_instance["lightId"] = np.array(len(self.lights), dtype=np.int32)
                light = {"type": "area", "shape_data": shape, "emission": material_parameter.emission,
                         "isTwosided": material_parameter.is_double_sided}
                self.lights.append(light)
            else:
                if material_parameter.diffuse_map:
                    material_parameter.diffuse_map_id = self.texture_name_id_dictionary[material_parameter.diffuse_map]

                if material_parameter.is_cutoff:
                    target_material = cutout_material
                else:
                    target_material = opaque_material
                if material_parameter.color0 is not None:
                    geometry_instance = GeometryInstance(geometry, target_material)
                    geometry_instance['programId'] = np.array(0, dtype=np.int32)
                    geometry_instance['hasCheckerboard'] = np.array(1, dtype=np.int32)
                    geometry_instance['color0'] = material_parameter.color0
                    geometry_instance['color1'] = material_parameter.color1
                    geometry_instance['to_uv'] = material_parameter.to_uv
                    geometry_instance['materialId'] = np.array(material_parameter.uuid, dtype=np.int32)
                elif material_parameter.type == "diffuse":
                    geometry_instance = GeometryInstance(geometry, target_material)
                    geometry_instance['programId'] = np.array(0, dtype=np.int32)
                    geometry_instance['materialId'] = np.array(material_parameter.uuid, dtype=np.int32)
                elif material_parameter.type == "dielectric":
                    geometry_instance = GeometryInstance(geometry, target_material)
                    geometry_instance['programId'] = np.array(1, dtype=np.int32)
                    geometry_instance['materialId'] = np.array(material_parameter.uuid, dtype=np.int32)
                elif material_parameter.type == "roughdielectric":
                    geometry_instance = GeometryInstance(geometry, target_material)
                    geometry_instance['programId'] = np.array(2, dtype=np.int32)
                    geometry_instance['materialId'] = np.array(material_parameter.uuid, dtype=np.int32)
                elif material_parameter.type == "conductor":
                    geometry_instance = GeometryInstance(geometry, target_material)
                    geometry_instance['programId'] = np.array(3, dtype=np.int32)
                    geometry_instance['eta'] = material_parameter.eta
                    geometry_instance['k'] = material_parameter.k
                    geometry_instance['materialId'] = np.array(material_parameter.uuid, dtype=np.int32)
                elif material_parameter.type == "roughconductor":
                    geometry_instance = GeometryInstance(geometry, target_material)
                    geometry_instance['programId'] = np.array(4, dtype=np.int32)
                    geometry_instance['eta'] = material_parameter.eta
                    geometry_instance['k'] = material_parameter.k
                    geometry_instance['materialId'] = np.array(material_parameter.uuid, dtype=np.int32)
                elif material_parameter.type == "plastic":
                    geometry_instance = GeometryInstance(geometry, target_material)
                    geometry_instance['programId'] = np.array(5, dtype=np.int32)
                    geometry_instance['materialId'] = np.array(material_parameter.uuid, dtype=np.int32)
                elif material_parameter.type == "roughplastic":
                    geometry_instance = GeometryInstance(geometry, target_material)
                    geometry_instance['programId'] = np.array(6, dtype=np.int32)
                    geometry_instance['materialId'] = np.array(material_parameter.uuid, dtype=np.int32)
                elif material_parameter.type == "disney":
                    geometry_instance = GeometryInstance(geometry, target_material)
                    geometry_instance['programId'] = np.array(99, dtype=np.int32)
                    geometry_instance['materialId'] = np.array(material_parameter.uuid, dtype=np.int32)
                if force_all_diffuse:
                    geometry_instance['programId'] = np.array(0, dtype=np.int32)

            if shape.transformation is not None:
                bbox = get_bbox_transformed(bbox, shape.transformation)

            # merge bbox
            self.bbox = get_bbox_merged(self.bbox, bbox)

            if shape_type == "obj":
                geometry_instance["faceNormals"] = np.array(1 if shape.face_normals else 0, dtype=np.int32)
            transform = add_transform(shape.transformation, geometry_instance)
            #if shape.transformation is not None:
            # geometry_instance["transformation"] = shape.transformation
            if material_parameter.type == "light":
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

    def optix_create_objs(self, program_dictionary):
        mesh_bb = program_dictionary['tri_mesh_bb']
        mesh_it = program_dictionary['tri_mesh_it']

        for obj_file_name in self.obj_name_list:
            mesh = OptixMesh(mesh_bb, mesh_it)
            mesh.load_from_file(self.folder_path + "/" + obj_file_name)
            self.obj_geometry_dict[obj_file_name] = mesh

    def optix_load_images(self):
        print("--------------- Texture List ---------------")
        print(self.texture_name_list)
        i = 1
        for texture_name in self.texture_name_list:
            if texture_name.endswith(".exr"):
                image = load_exr_image(self.folder_path + "/" + texture_name)
            elif texture_name.endswith(".pfm"):
                image = load_exr_image(self.folder_path + "/" + texture_name)
            elif texture_name.endswith(".hdr"):
                image = load_exr_image(self.folder_path + "/" + texture_name, True)
            else:
                image = Image.open(self.folder_path + "/" + texture_name).convert('RGBA')
            image_np = np.asarray(image)
            tex_buffer = Buffer.from_array(image_np, buffer_type='i', drop_last_dim=True)

            tex_sampler = TextureSampler(tex_buffer,
                                         wrap_mode='repeat',
                                         indexing_mode='normalized_coordinates',
                                         read_mode='normalized_float',
                                         filter_mode='linear')

            self.texture_name_id_dictionary[texture_name] = tex_sampler.get_id()
            print(texture_name, tex_sampler.get_buffer().get_size())
            print(texture_name, i)
            print(texture_name, tex_sampler.get_id())
            print(texture_name, tex_sampler.get_buffer().dtype)
            self.texture_sampler_list.append(tex_sampler)
            i += 1

    def load_shapes(self, root):
        shape_list = []
        obj_list = []
        annonymous_material_count = 0

        for shape in root.findall('shape'):
            # 1. load shape
            shape_parameter = load_single_shape(shape)
            if shape_parameter.obj_file_name is not None and shape_parameter.obj_file_name not in obj_list:
                obj_list.append(shape_parameter.obj_file_name)

            # 2. load material
            material_ref = shape.find("ref")
            if material_ref is not None:
                bsdf_id = material_ref.attrib["id"]
                if bsdf_id in self.material_name_to_id_dictionary:
                    bsdf_index = self.material_name_to_id_dictionary[bsdf_id]
                    material_parameter = self.material_list[bsdf_index]
                else:
                    bsdf = root.find('bsdf[@id="%s"]' % bsdf_id)
                    material_parameter = self.load_new_material(bsdf)
            else:
                bsdf_id = "annonymous_material_%d" % annonymous_material_count
                annonymous_material_count += 1
                bsdf = shape.find("bsdf")
                material_parameter = self.load_new_material(bsdf, bsdf_id=bsdf_id)

            # 3. load emitter
            emitter = shape.find("emitter")
            if emitter is not None:
                spectrum_radiance = emitter.find('spectrum[@name="radiance"]')
                rgb_radiance = emitter.find('rgb[@name="radiance"]')
                if spectrum_radiance is not None:
                    radiance = spectrum_radiance.attrib['value']
                if rgb_radiance is not None:
                    radiance = rgb_radiance.attrib['value']
                radiance = str2floatarray(radiance)
                material_parameter_light = MaterialParameter("light_")
                material_parameter_light.type = 'light'
                material_parameter_light.is_double_sided = material_parameter.is_double_sided
                material_parameter_light.emission = radiance
                material_parameter = material_parameter_light
            # else:
            #     # 3. load material
            #     material_ref = shape.find("ref")
            #     if material_ref is not None:
            #         bsdf_id = material_ref.attrib["id"]
            #         if bsdf_id in self.material_name_to_id_dictionary:
            #             bsdf_index = self.material_name_to_id_dictionary[bsdf_id]
            #             material_parameter = self.material_list[bsdf_index]
            #         else:
            #             bsdf = root.find('bsdf[@id="%s"]' % bsdf_id)
            #             material_parameter = self.load_new_material(bsdf)
            #     else:
            #         bsdf_id = "annonymous_material_%d" % annonymous_material_count
            #         annonymous_material_count += 1
            #         bsdf = shape.find("bsdf")
            #         material_parameter = self.load_new_material(bsdf, bsdf_id=bsdf_id)

            shape_parameter.material_parameter = material_parameter
            shape_list.append(shape_parameter)

        # resolve cube
        new_shape_list = []
        for shape in shape_list:
            if shape.shape_type == "cube":
                for i in range(6):
                    shape_parameter = ShapeParameter()
                    shape_parameter.shape_type = "rectangle"
                    shape_parameter.rectangle_info = shape.rectangle_info[i]
                    shape_parameter.material_parameter = shape.material_parameter
                    new_shape_list.append(shape_parameter)
            else:
                new_shape_list.append(shape)

        self.shape_list = new_shape_list
        self.obj_name_list = obj_list

    def load_new_material(self, bsdf, bsdf_id=None):
        # load material information
        material = load_single_material(bsdf, bsdf_id=bsdf_id)

        # append material to material list
        material.uuid = len(self.material_list)
        self.material_name_to_id_dictionary[material.name] = len(self.material_list)
        self.material_list.append(material)

        # append texture
        if material.diffuse_map is not None:
            self.texture_name_list.append(material.diffuse_map)

        return material
