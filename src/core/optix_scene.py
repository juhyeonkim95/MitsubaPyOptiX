from core.scene import Scene
from pyoptix import Context, Program, Material, Group, Acceleration, Buffer
import numpy as np
from core.textures.texture import Texture
from core.bsdfs.bsdf import BSDF
from core.emitters.emitter import Emitter


class OptiXSceneContext:
    def __init__(self, context: Context):
        """
        This is never changed during rendering same scene.
        """
        self.context = context

        self.program_dictionary = {}
        self.material_dict = {}

        # Common optix context setting/programs/materials
        self.init_optix_context()
        self.init_optix_programs()
        self.init_optix_materials()

    def load_scene(self, scene: Scene):
        scene_epsilon = 1e-3 if scene.name == "veach_door_simple" else 1e-5
        self.context['scene_epsilon'] = np.array(scene_epsilon, dtype=np.float32)

        # scene specific
        self.init_entry_point(scene.has_envmap)
        self.init_optix_scene(scene)
        self.init_camera(scene)

    def init_optix_context(self):
        context = self.context
        context.set_ray_type_count(2)
        context.set_entry_point_count(2)

        context['pathtrace_ray_type'] = np.array(0, dtype=np.uint32)
        context['pathtrace_shadow_ray_type'] = np.array(1, dtype=np.uint32)
        context['bad_color'] = np.array([1000000., 0., 1000000.], dtype=np.float32)
        context['bg_color'] = np.zeros(3, dtype=np.float32)

    def init_optix_programs(self):
        program_dictionary = {}
        # renderer
        target_program = 'optix/programs/path_trace_camera.cu'
        program_dictionary["ray_generation"] = Program(target_program, 'pathtrace_camera')
        program_dictionary["exception"] = Program(target_program, 'exception')

        program_dictionary["miss_envmap"] = Program('optix/programs/miss_program.cu', 'miss_environment_mapping')
        program_dictionary["miss"] = Program('optix/programs/miss_program.cu', 'miss')

        # Geometries
        program_dictionary['tri_mesh_bb'] = Program('optix/shapes/triangle_mesh.cu', 'mesh_bounds')
        program_dictionary['tri_mesh_it'] = Program('optix/shapes/triangle_mesh.cu', 'mesh_intersect_refine')

        program_dictionary["quad_bb"] = Program('optix/shapes/parallelogram.cu', 'bounds')
        program_dictionary["quad_it"] = Program('optix/shapes/parallelogram.cu', 'intersect')
        program_dictionary["sphere_bb"] = Program('optix/shapes/sphere.cu', 'bounds')
        program_dictionary["sphere_it"] = Program('optix/shapes/sphere.cu', 'robust_intersect')
        program_dictionary["disk_bb"] = Program('optix/shapes/disk.cu', 'bounds')
        program_dictionary["disk_it"] = Program('optix/shapes/disk.cu', 'intersect')
        program_dictionary["box_bb"] = Program('optix/shapes/box.cu', 'box_bounds')
        program_dictionary["box_it"] = Program('optix/shapes/box.cu', 'box_intersect')

        # Materials
        closest_hit_program = 'optix/programs/hit_program.cu'
        any_hit_hit_program = 'optix/programs/any_hit_program.cu'

        program_dictionary["closest_hit"] = Program(closest_hit_program, 'closest_hit')
        program_dictionary["closest_hit_light"] = Program('optix/programs/light_hit_program.cu', 'diffuseEmitter')
        program_dictionary["any_hit_cutout"] = Program(any_hit_hit_program, 'any_hit_cutout')

        program_dictionary["any_hit_shadow"] = Program(any_hit_hit_program, 'any_hit_shadow')
        program_dictionary["any_hit_shadow_cutout"] = Program(any_hit_hit_program, 'any_hit_shadow_cutout')

        program_dictionary['quad_tree_updater'] = Program('optix/radiance_record/quad_tree_updater.cu', 'quad_tree_updater')
        program_dictionary['binary_tree_updater'] = Program('optix/radiance_record/binary_tree_updater.cu', 'spatial_binary_tree_updater')

        self.program_dictionary = program_dictionary

    def init_optix_materials(self):
        material_dict = {}
        program_dictionary = self.program_dictionary

        opaque_material = Material()
        opaque_material.set_closest_hit_program(0, program_dictionary["closest_hit"])
        opaque_material.set_any_hit_program(1, program_dictionary["any_hit_shadow"])

        cutout_material = Material()
        cutout_material.set_closest_hit_program(0, program_dictionary["closest_hit"])
        cutout_material.set_any_hit_program(0, program_dictionary["any_hit_cutout"])
        cutout_material.set_any_hit_program(1, program_dictionary["any_hit_shadow_cutout"])

        light_material = Material()
        light_material.set_closest_hit_program(0, program_dictionary["closest_hit_light"])
        light_material.set_any_hit_program(1, program_dictionary["any_hit_shadow"])

        material_dict['opaque_material'] = opaque_material
        material_dict['cutout_material'] = cutout_material
        material_dict['light_material'] = light_material
        self.material_dict = material_dict

    def init_entry_point(self, has_envmap):
        if has_envmap:
            miss_program = self.program_dictionary["miss_envmap"]
        else:
            miss_program = self.program_dictionary["miss"]

        self.context.set_ray_generation_program(0, self.program_dictionary['ray_generation'])
        self.context.set_exception_program(0, self.program_dictionary['exception'])
        self.context.set_miss_program(0, miss_program)

        self.context.set_ray_generation_program(1, self.program_dictionary["quad_tree_updater"])

    def init_optix_scene(self, scene:Scene):
        """
        Init Optix related things to the scene
        :return:
        """
        context = self.context

        # (1) load texture data to optix and retrieve optix texture ids
        scene.optix_load_textures()
        # (2) load OBJ mesh data to optix and retrieve optix buffer ids
        scene.optix_load_objs(self.program_dictionary)
        # (3) from (1) and (2), create optix geometry instances
        scene.optix_create_geometry_instances(self.program_dictionary, self.material_dict, False)

        # (4) Assign optix objects to context
        self.load_scene_geometry_group(scene)
        self.load_scene_lights(scene)
        self.load_scene_materials(scene)

    def load_scene_geometry_group(self, scene: Scene):
        shadow_group = Group(children=scene.geometry_instances)
        shadow_group.set_acceleration(Acceleration("Trbvh"))
        self.context['top_shadower'] = shadow_group

        group = Group(children=(scene.geometry_instances + scene.light_instances))
        group.set_acceleration(Acceleration("Trbvh"))
        self.context['top_object'] = group

    def load_scene_lights(self, scene: Scene):
        np_lights = np.array([np.array(x) for x in scene.light_list])
        self.context["sysLightParameters"] = \
            Buffer.from_array(np_lights, dtype=Emitter.dtype, buffer_type='i', drop_last_dim=True)

    def load_scene_materials(self, scene: Scene):
        np_materials = np.array([np.array(x) for x in scene.material_list])
        self.context["sysMaterialParameters"] = \
            Buffer.from_array(np_materials, dtype=BSDF.dtype, buffer_type='i', drop_last_dim=True)

        if len(scene.texture_list) > 0:
            np_textures = np.array([np.array(x) for x in scene.texture_list])
            self.context["sysTextureParameters"] = \
                Buffer.from_array(np_textures, dtype=Texture.dtype, buffer_type='i', drop_last_dim=True)
        else:
            self.context["sysTextureParameters"] =\
                Buffer.empty((1, 1), Texture.dtype, buffer_type='i', drop_last_dim=True)

    def init_camera(self, scene):
        aspect_ratio = float(scene.width) / float(scene.height)
        camera = scene.camera
        u, v, w = camera.calc_image_space_vectors(aspect_ratio)

        # upload to context
        context = self.context
        context["eye"] = np.array(camera.eye, dtype=np.float32)
        context["U"] = np.array(u, dtype=np.float32)
        context["V"] = np.array(v, dtype=np.float32)
        context["W"] = np.array(w, dtype=np.float32)
        context["focalDistance"] = np.array(5, dtype=np.float32)
        # context["apertureRadius"] = np.array(0.5, dtype=np.float32)
        context["camera_type"] = np.array(0, dtype=np.uint32)
