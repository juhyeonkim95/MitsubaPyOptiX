from core.camera import *
from core.raycasting.common_structs import *
from os.path import dirname
import time
from datetime import timedelta
from main import set_camera

height = 512
width = 512
from pyoptix import Buffer, Context, Compiler, EntryPoint, Program, GeometryGroup, Acceleration
from core.scene import *
import matplotlib.pyplot as plt

Compiler.add_program_directory(dirname(__file__))


def create_context():
    context = Context()
    context.set_ray_type_count(1)
    context.set_entry_point_count(1)
    context.set_stack_size(2000)

    entry_point = EntryPoint(Program('optix/raycasting/ray_gen_raycast.cu', 'ray_gen'))

    return context, entry_point


def create_rays(camera: Camera, rays_arrays):
    fov = camera.fov
    aspect_ratio = float(width) / float(height)
    fovx = camera.fov_axis == 'x'

    # claculate camera variables
    W = np.array(camera.w)
    U = np.array(camera.u)
    V = np.array(camera.v)
    wlen = np.sqrt(np.sum(W ** 2))
    if fovx:
        ulen = wlen * math.tan(0.5 * fov * math.pi / 180)
        U *= ulen
        vlen = ulen / aspect_ratio
        V *= vlen
    else:
        vlen = wlen * math.tan(0.5 * fov * math.pi / 180)
        V *= vlen
        ulen = vlen * aspect_ratio
        U *= ulen

    for y in range(height):
        for x in range(width):
            dx = x / width * 2 - 1
            dy = y / height * 2 - 1
            rays_arrays[y, x]["tmin"] = 0.0
            rays_arrays[y, x]["tmax"] = 1e30
            rays_arrays[y, x]["dir"] = normalize(dx * U + dy * V + W)
            rays_arrays[y, x]["origin"] = camera.eye


def update_rays(context, hits_arrays, rays_arrays):
    context['hits'].copy_to_array(hits_arrays)
    context['rays'].copy_to_array(rays_arrays)
    rays_arrays["dir"] = hits_arrays["new_direction"]
    rays_arrays["origin"] = hits_arrays["hit_point"]
    context['rays'] = Buffer.from_array(rays_arrays, dtype=RayData.dtype, buffer_type='io')


def create_object_instances(scene):
    par_bb = Program('optix/shapes/parallelogram.cu', 'bounds')
    par_int = Program('optix/shapes/parallelogram.cu', 'intersect')
    sph_bb = Program('optix/shapes/sphere.cu', 'bounds')
    sph_int = Program('optix/shapes/sphere.cu', 'intersect')
    disk_bb = Program('optix/shapes/disk.cu', 'bounds')
    disk_int = Program('optix/shapes/disk.cu', 'intersect')

    closest_hit = Material(closest_hit={0: Program('optix/raycasting/hit_program_raycast.cu', 'closest_hit')})
    closest_hit_emitter = Material(
        closest_hit={0: Program('optix/raycasting/light_hit_program_raycast.cu', 'closest_hit')})
    closest_hit["programId"] = np.array(0, dtype=np.int32)

    for shape in scene.shape_list:
        shape_type = shape.shape_type
        geometry = None
        if shape_type == "obj":
            mesh = scene.obj_geometry_dict[shape.obj_file_name]
            geometry = mesh.geometry
        elif shape_type == "rectangle":
            o, u, v = shape.rectangle_info
            geometry = create_parallelogram(o, u, v, par_int, par_bb)
        elif shape_type == "sphere":
            geometry = create_sphere(shape.center, shape.radius, sph_int, sph_bb)
        elif shape_type == "disk":
            geometry = create_disk(shape.center, shape.radius, shape.normal, disk_int, disk_bb)
        material_parameter = shape.material_parameter

        if material_parameter.type == "light":
            geometry_instance = GeometryInstance(geometry, closest_hit_emitter)
            geometry_instance["emission_color"] = material_parameter.emission
        else:
            geometry_instance = GeometryInstance(geometry, closest_hit)
            geometry_instance["diffuse_color"] = material_parameter.color
            geometry_instance["diffuse_map_id"] = np.array(0, dtype=np.int32)

        if shape.transformation is not None:
            geometry_instance["transformation"] = shape.transformation
        scene.geometry_instances.append(geometry_instance)


def create_geometry(context, scene):
    geometry_instances = scene.geometry_instances
    group = GeometryGroup(children=geometry_instances)
    group.set_acceleration(Acceleration("Trbvh"))
    context['top_object'] = group


def render(scene_name):
    global width, height
    context, entry_point = create_context()
    scene = Scene(scene_name)
    scene.load_scene_from("../data/%s/scene.xml" % scene_name)
    scene.load_images()
    scene.create_objs()
    width = scene.width
    height = scene.height
    rays_array = np.empty((height, width), dtype=RayData.dtype)
    hits_array = np.empty((height, width), dtype=Hit.dtype)
    context['hits'] = Buffer.from_array(hits_array, dtype=Hit.dtype, buffer_type='io')
    context['rays'] = Buffer.from_array(rays_array, dtype=RayData.dtype, buffer_type='io')

    create_object_instances(scene)
    create_geometry(context, scene)
    print("Start launch")
    initialized = False
    total_radiance = np.zeros((width, height, 3), dtype=np.float32)
    start_time = time.time()
    SPP = 32
    set_camera(context, scene.camera)

    for i in range(SPP):
        print(i)
        context["frame_number"] = np.array((i + 1), dtype=np.uint32)
        for depth in range(8):
            context["depth"] = np.array(depth, dtype=np.uint32)
            if not initialized:
                entry_point.launch((width, height))
                initialized = True
            else:
                context.launch(0, width, height)
            update_rays(context, hits_array, rays_array)
        prd_array = context['hits'].to_array()
        radiance = np.reshape(prd_array["result"], (width, height, 3))
        total_radiance += radiance
    total_radiance /= SPP

    end_time = time.time()
    elasped_time = end_time - start_time
    print("Elapsed Time:", str(timedelta(seconds=elasped_time)))

    image = LinearToSrgb(ToneMap(total_radiance, 1.5))
    plt.imshow(image, origin='lower')
    plt.show()


if __name__ == '__main__':
    render("cornell_box")
