from pyoptix import Group, Acceleration, Buffer
from core.textures.texture import Texture
from core.utils.math_utils import *
from utils.timing_utils import *
from core.bsdfs.bsdf import BSDF
from core.emitters.emitter import Emitter



@timing
def create_geometry(context, scene):
    geometry_instances = scene.geometry_instances
    light_instances = scene.light_instances

    shadow_group = Group(children=geometry_instances)
    shadow_group.set_acceleration(Acceleration("Trbvh"))
    context['top_shadower'] = shadow_group

    group = Group(children=(geometry_instances + light_instances))
    group.set_acceleration(Acceleration("Trbvh"))
    context['top_object'] = group


@timing
def create_scene_lights(context, scene):
    np_lights = np.array([np.array(x) for x in scene.light_list])
    print("Light list", np_lights)
    light_buffer = Buffer.from_array(np_lights, dtype=Emitter.dtype, buffer_type='i', drop_last_dim=True)
    context["sysLightParameters"] = light_buffer

    # lights = []
    # for light_data in scene.lights:
    #     if light_data["type"] == "envmap":
    #         light_data["envmapID"] = scene.texture_name_to_optix_index_dictionary[light_data["envmap_file"]]
    #         print("Envmap ID selected!", light_data["envmapID"])
    #     light = Light(light_data)
    #     print(light)
    #     lights.append(np.array(light))
    # np_l = np.array(lights)
    # light_buffer = Buffer.from_array(np_l, dtype=Light.dtype, buffer_type='i', drop_last_dim=True)
    # context["lights"] = light_buffer


@timing
def create_scene_materials(context, scene):
    materials = []
    for m in scene.material_list:
        m_np = np.array(m)
        materials.append(m_np)

    textures = []
    for t in scene.texture_list:
        textures.append(np.array(t))

    np_materials = np.array([np.array(x) for x in scene.material_list])
    print("Material list", np_materials)

    context["sysMaterialParameters"] = Buffer.from_array(np_materials, dtype=BSDF.dtype, buffer_type='i', drop_last_dim=True)
    if len(scene.texture_list) > 0:
        np_textures = np.array([np.array(x) for x in scene.texture_list])
        context["sysTextureParameters"] = Buffer.from_array(np_textures, dtype=Texture.dtype, buffer_type='i', drop_last_dim=True)
    else:
        context["sysTextureParameters"] = Buffer.empty((1, 1), Texture.dtype, buffer_type='i', drop_last_dim=True)


def create_q_table_related(context, room_size, height, width, N_CUBE, UV_N, octree):
    # input_array = np.zeros((UV_N * UV_N * 2, 3), dtype=np.float32)
    # for i in range(2 * UV_N * UV_N):
    #     v = getDirectionFrom(i, (0.5, 0.5), (UV_N, UV_N))
    #     input_array[i][0] = v[0]
    #     input_array[i][1] = v[1]
    #     input_array[i][2] = v[2]
    #
    # context['unitUVVectors'] = Buffer.from_array(input_array, dtype=np.float32, buffer_type='i', drop_last_dim=True)
    # context['hit_count_buffer'] = Buffer.empty((height, width, 1), dtype=np.float32, buffer_type='o', drop_last_dim=True)
    # context['path_length_buffer'] = Buffer.empty((height, width, 1), dtype=np.float32, buffer_type='o', drop_last_dim=True)
    # context['scatter_type_buffer'] = Buffer.empty((height, width, 2), dtype=np.float32, buffer_type='o', drop_last_dim=True)

    unit_cube_number = np.array([N_CUBE, N_CUBE, N_CUBE], dtype=np.uint32)
    unit_cube_size = room_size / unit_cube_number.astype(np.float32)

    sphere_uv_map_number = np.array([UV_N, UV_N], dtype=np.uint32)

    if octree is None:
        state_number = int(np.prod(unit_cube_number))
    else:
        state_number = octree.node_number
    action_number = int(np.prod(sphere_uv_map_number)) * 2

    print("Total State Number", state_number, action_number)

    q_table_init = np.zeros((action_number, state_number), dtype=np.float32)
    q_table_init.fill(1e-3)

    policy_table_init = np.zeros((action_number, state_number), dtype=np.float32)
    policy_table_init.fill(1e-3)

    normal_table_init = np.zeros((3, state_number), dtype=np.float32)
    normal_table_init.fill(1e-3)
    normal_table_init2 = np.zeros((3, state_number), dtype=np.float32)
    normal_table_init2.fill(1e-3)

    #visit_counts = np.zeros((action_number, state_number), dtype=np.uint32)
    #mcmc_init = np.random.random((action_number, state_number, 2)).astype(np.float32)
    #print(mcmc_init.dtype)
    #context['q_table_accumulated'] = Buffer.empty((action_number, state_number), dtype=np.float32, buffer_type='io', drop_last_dim=False)
    #context['irradiance_table'] = Buffer.empty((action_number, state_number), dtype=np.float32, buffer_type='io', drop_last_dim=False)
    #context['max_radiance_table'] = Buffer.empty((action_number, state_number), dtype=np.float32, buffer_type='io', drop_last_dim=False)

    #context['q_table'] = Buffer.from_array(q_table_init, dtype=np.float32, buffer_type='io', drop_last_dim=False)
    #context['q_table_old'] = Buffer.from_array(policy_table_init, dtype=np.float32, buffer_type='io', drop_last_dim=False)
    #context['normal_table'] = Buffer.from_array(normal_table_init, dtype=np.float32, buffer_type='io', drop_last_dim=False)
    #context['normal_table_old'] = Buffer.from_array(normal_table_init2, dtype=np.float32, buffer_type='io', drop_last_dim=False)
    #context['mcmc_table'] = Buffer.from_array(mcmc_init, dtype=np.float32, buffer_type='io', drop_last_dim=True)

    # if octree is None:
    #     context['stree_visit_count'] = Buffer.empty((0, ), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
    #     context['stree_index_array'] = Buffer.empty((0, ), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
    #     context['stree_rank_array'] = Buffer.empty((0, ), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
    # else:
    #     context['stree_visit_count'] = Buffer.empty((octree.total_size, ), dtype=np.uint32, buffer_type='i', drop_last_dim=False)
    #     context['stree_index_array'] = Buffer.from_array(octree.index_array, dtype=np.uint32, buffer_type='i', drop_last_dim=False)
    #     context['stree_rank_array'] = Buffer.from_array(octree.rank_array, dtype=np.uint32, buffer_type='i', drop_last_dim=False)

    #context['visit_counts'] = Buffer.from_array(visit_counts, dtype=np.float32, buffer_type='io', drop_last_dim=False)
    #print("Unit cube size", unit_cube_size)
    #print("unitUVNumber", sphere_uv_map_number)

    #context['unitCubeSize'] = unit_cube_size
    #context['unitCubeNumber'] = unit_cube_number
    #context['unitUVNumber'] = sphere_uv_map_number
