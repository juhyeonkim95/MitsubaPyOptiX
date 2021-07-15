from core.camera import Camera
from core.parameters import MaterialParameter, ShapeParameter
from core.utils.math_utils import *
from utils.image_utils import *
import re


def str_to_bool(s):
    if s=="true" or s=="True":
        return True
    elif s=="false" or s=="False":
        return False
    return False


def find_bool(node, name, default=False):
    if node.find(name) is None:
        return default
    else:
        value = node.find(name).attrib["value"]
        return str_to_bool(value)


def find_xyz(node, name, default=None):
    if node.find(name) is not None:
        x = float(node.find(name).attrib["x"])
        y = float(node.find(name).attrib["y"])
        z = float(node.find(name).attrib["z"])
        value = [x, y, z]
    else:
        value = default
    return np.array(value, dtype=np.float32)


def find(node, name, default=None):
    if node.find(name) is not None:
        value = node.find(name).attrib["value"]
    else:
        if type(default) == str:
            return default
        elif type(default) == float:
            return default
        elif type(default) == list:
            return np.array(default, dtype=np.float32)
        value = default
    if type(value) is float:
        return value

    values = re.split(",| ", value)
    values = [v for v in values if len(v) > 0]
    if len(values) > 1:
        float_array = [float(v) for v in values]
        return np.array(float_array, dtype=np.float32)
    else:
        if type(default) == str:
            return value
        elif type(default) == bool:
            return str_to_bool(value)
        else:
            return float(value)


def str2floatarray(s):
    s = s.replace(",", "")
    fs = s.split(" ")
    float_array = [float(x) for x in fs]
    return np.array(float_array, dtype=np.float32)


def str2_4by4mat(s):
    ms = str2floatarray(s)
    #ms = np.array(fs)
    ms = ms.reshape((4, 4))
    return ms


def rectangle(mat):
    original_point = np.array([-1., -1., 0., 1.], dtype=np.float32)
    unit_x = np.array([1., -1., 0., 1.], dtype=np.float32)
    unit_y = np.array([-1., 1., 0., 1.], dtype=np.float32)

    anchor = mat.dot(original_point)[0:3]
    offset1 = mat.dot(unit_x)[0:3] - anchor
    offset2 = mat.dot(unit_y)[0:3] - anchor

    return anchor, offset1, offset2


def rectangle_from(o, dx, dy, mat):
    anchor = mat.dot(o)[0:3]
    offset1 = mat.dot(o+dx)[0:3] - anchor
    offset2 = mat.dot(o+dy)[0:3] - anchor
    return anchor, offset1, offset2


def cube(mat):
    o1 = np.array([-1., -1., -1., 1.], dtype=np.float32)
    vx = np.array([2., 0., 0., 0.], dtype=np.float32)
    vy = np.array([0., 2., 0., 0.], dtype=np.float32)
    vz = np.array([0., 0., 2., 0.], dtype=np.float32)
    r1 = rectangle_from(o1, vy, vx, mat)
    r2 = rectangle_from(o1, vz, vy, mat)
    r3 = rectangle_from(o1, vx, vz, mat)

    o1 = np.array([1., 1., 1., 1.], dtype=np.float32)
    vx = np.array([-2., 0., 0., 0.], dtype=np.float32)
    vy = np.array([0., -2., 0., 0.], dtype=np.float32)
    vz = np.array([0., 0., -2., 0.], dtype=np.float32)
    r4 = rectangle_from(o1, vx, vy, mat)
    r5 = rectangle_from(o1, vy, vz, mat)
    r6 = rectangle_from(o1, vz, vx, mat)

    return [r1, r2, r3, r4, r5, r6]


def load_camera(sensor):
    camera_fov = float(sensor.find('float[@name="fov"]').attrib['value'])
    camera = Camera(camera_fov)

    camera_transform = sensor.find('transform[@name="toWorld"]')

    if camera_transform.find("matrix") is not None:
        camera_transform_matrix = str2_4by4mat(camera_transform.find("matrix").attrib['value'])
        camera.load_from_matrix(camera_transform_matrix)
    elif camera_transform.find("lookat") is not None:
        lookat_attrib = camera_transform.find("lookat").attrib
        camera_lookat = str2floatarray(lookat_attrib["target"])
        camera_origin = str2floatarray(lookat_attrib["origin"])
        camera_up = str2floatarray(lookat_attrib["up"])
        camera.load_from_lookat(camera_lookat, camera_origin, camera_up)

    # Print camera info
    print(camera)

    return camera


def load_scene_size(sensor):
    height = int(sensor.find('film/integer[@name="height"]').attrib['value'])
    width = int(sensor.find('film/integer[@name="width"]').attrib['value'])
    return height, width


def transform_point(to_world, p):
    p = np.array([p[0], p[1], p[2], 1], dtype=np.float32)
    p = np.matmul(to_world, p)
    p = p[0:3]
    return p


def transform_vector(to_world, v):
    v = np.array([v[0], v[1], v[2], 0], dtype=np.float32)
    v = np.matmul(to_world, v)
    v = v[0:3]
    return v


def get_pos_dir(node):
    to_world = node.find("transform/matrix").attrib['value']
    to_world = str2_4by4mat(to_world)
    position = transform_point(to_world, [0, 0, 0])
    direction = transform_vector(to_world, [0, 0, 1])
    direction = normalize(direction)
    return position, direction


def get_pos_dir_rad(node):
    to_world = node.find("transform/matrix").attrib['value']
    to_world = str2_4by4mat(to_world)
    position = transform_point(to_world, [0, 0, 0])
    direction = transform_vector(to_world, [0, 0, 1])
    direction = normalize(direction)
    x_axis = transform_point(to_world, [1, 0, 0])
    radius = length(x_axis - position)
    return position, direction, radius


def load_emitter(root, folder_path):
    lights = []
    env_map_file = None

    for emitter in root.findall('emitter'):
        emitter_type = emitter.attrib['type']
        light_info = {}
        if emitter_type == "point":
            to_world = str2_4by4mat(emitter.find("transform/matrix").attrib['value'])
            position = transform_point(to_world, [0, 0, 0])
            intensity = emitter.find('rgb[@name="intensity"]').attrib["value"]
            intensity = str2floatarray(intensity)
            light_info = {"type": "point", "intensity": intensity, "position": position}
            lights.append(light_info)
        elif emitter_type == "spot":
            position, direction = get_pos_dir(emitter)
            intensity = emitter.find('rgb[@name="intensity"]').attrib["value"]
            intensity = str2floatarray(intensity)
            cutoffAngle = float(emitter.find('float[@name="cutoffAngle"]').attrib["value"])
            if emitter.find('float[@name="beamWidth"]') is not None:
                beamWidth = float(emitter.find('float[@name="beamWidth"]').attrib["value"])
            else:
                beamWidth = 0.75 * cutoffAngle
            light_info = {"type": "spot", "intensity": intensity, "position": position, "direction": direction,
                          "cutoffAngle": cutoffAngle, "beamWidth": beamWidth}
            lights.append(light_info)
        elif emitter_type == "directional":
            light_info["type"] = "directional"
            light_info["direction"] = normalize(find_xyz(emitter, 'vector[@name="direction"]', [0, 1.0, 0]))
            light_info['emission'] = find(emitter, 'rgb[@name="irradiance"]', [1, 1, 1])
            lights.append(light_info)
        elif emitter_type == "envmap":
            to_world = str2_4by4mat(emitter.find("transform/matrix").attrib['value'])
            env_map_file = emitter.find('string[@name="filename"]').attrib["value"]
            light_info["type"] = "envmap"
            light_info["envmap_file"] = env_map_file
            light_info["transformation"] = to_world
            lights.append(light_info)
        elif emitter_type == "sunsky":
            #env_map_file = "../common/textures/Sky 19.exr"
            light_info["type"] = "directional"
            #light_info["envmap_file"] = env_map_file
            #light_info["transformation"] = np.eye(4, dtype=np.float32)
            light_info["direction"] = find_xyz(emitter, 'vector[@name="sunDirection"]', [0, 1.0, 0])
            light_info['emission'] = find(emitter, 'rgb[@name="intensity"]', [30, 30, 30])
            lights.append(light_info)

    return lights, env_map_file


def load_single_shape(shape):
    shape_type = shape.attrib['type']

    shape_parameter = ShapeParameter()
    shape_parameter.shape_type = shape_type

    if shape_type == "rectangle":
        to_world = shape.find("transform/matrix").attrib['value']
        to_world = str2_4by4mat(to_world)
        shape_parameter.rectangle_info = rectangle(to_world)
    elif shape_type == "cube":
        to_world = shape.find("transform/matrix").attrib['value']
        to_world = str2_4by4mat(to_world)
        shape_parameter.rectangle_info = cube(to_world)
    elif shape_type == "sphere":
        if shape.find("transform/matrix") is not None:
            position, normal, radius = get_pos_dir_rad(shape)
            shape_parameter.center = position
            shape_parameter.radius = radius
        else:
            center = shape.find('point[@name="center"]')
            x = float(center.attrib["x"])
            y = float(center.attrib["y"])
            z = float(center.attrib["z"])
            shape_parameter.center = np.array([x, y, z], dtype=np.float32)
            shape_parameter.radius = float(shape.find('float[@name="radius"]').attrib["value"])
    elif shape_type == "disk":
        position, normal, radius = get_pos_dir_rad(shape)
        shape_parameter.center = position
        shape_parameter.radius = radius
        shape_parameter.normal = normal
    elif shape_type == "obj":
        mesh_file_name = shape.find('string[@name="filename"]').attrib["value"]
        face_normals = find_bool(shape, 'boolean[@name="faceNormals"]', False)
        transformation = shape.find('transform[@name="toWorld"]/matrix').attrib["value"]
        shape_parameter.transformation = str2_4by4mat(transformation)
        shape_parameter.obj_file_name = mesh_file_name
        shape_parameter.face_normals = face_normals

    return shape_parameter


def load_single_material(bsdf, bsdf_id=None):
    bsdf_type = bsdf.attrib['type']
    if 'id' in bsdf.attrib:
        bsdf_id = bsdf.attrib['id']
    material = MaterialParameter(bsdf_id)

    while True:
        if bsdf_type == "mask":
            if bsdf.find('rgb[@name="opacity"]') is not None:
                material.opacity = str2floatarray(bsdf.find('rgb[@name="opacity"]').attrib["value"])[0]
            material.is_cutoff = True
            bsdf = bsdf.find("bsdf")
            bsdf_type = bsdf.attrib['type']
        elif bsdf_type == "twosided":
            material.is_double_sided = True
            bsdf = bsdf.find("bsdf")
            bsdf_type = bsdf.attrib['type']
        elif bsdf_type == "bump" or bsdf_type == "bumpmap":
            bsdf = bsdf.find("bsdf")
            bsdf_type = bsdf.attrib['type']
        elif bsdf_type == "coating":
            bsdf = bsdf.find("bsdf")
            bsdf_type = bsdf.attrib['type']
        else:
            break

    # 1. Diffuse
    if bsdf_type == "diffuse":
        material.color = find(bsdf, 'rgb[@name="reflectance"]', [0.5, 0.5, 0.5])

    # 2. Dielectrics
    elif "dielectric" in bsdf_type:
        material.is_double_sided = False
        material.intIOR = find(bsdf, 'float[@name="intIOR"]', 1.5046)
        material.extIOR = find(bsdf, 'float[@name="extIOR"]', 1.000277)
        material.color = np.array([1, 1, 1], dtype=np.float32)

        if bsdf_type == "dielectric":
            material.roughness = 0
        elif bsdf_type == "roughdielectric":
            material.roughness = find(bsdf, 'float[@name="alpha"]', 0.1)
            material.distribution_type = find(bsdf, 'string[@name="distribution"]', "ggx")
        elif bsdf_type == "thindielectric":
            material.roughness = 0.0

    # 3. Conductors
    elif "conductor" in bsdf_type:
        material.metallic = 1.0
        material.eta = find(bsdf, 'rgb[@name="eta"]', [0, 0, 0])
        material.k = find(bsdf, 'rgb[@name="k"]', [1, 1, 1])
        material.color = np.array([1, 1, 1], dtype=np.float32)
        # material.color = find(bsdf, 'rgb[@name="specularReflectance"]', [1, 1, 1])

        if bsdf_type == "conductor":
            material.roughness = 0.0
        elif bsdf_type == "roughconductor":
            material.roughness = find(bsdf, 'float[@name="alpha"]', 0.5)
            material.distribution_type = find(bsdf, 'string[@name="distribution"]', "ggx")

    # 4. plastics
    elif "plastic" in bsdf_type:
        material.intIOR = find(bsdf, 'float[@name="intIOR"]', 1.5046)
        material.extIOR = find(bsdf, 'float[@name="extIOR"]', 1.000277)
        material.clearcoat = 1.0
        material.color = find(bsdf, 'rgb[@name="diffuseReflectance"]', [0.5, 0.5, 0.5])
        material.nonlinear = find(bsdf, 'boolean[@name="nonlinear"]', False)

        if bsdf_type == "plastic":
            material.roughness = 0
        elif bsdf_type == "roughplastic":
            material.roughness = find(bsdf, 'float[@name="alpha"]', 0.1)
            material.distribution_type = find(bsdf, 'string[@name="distribution"]', "ggx")

    else:
        print("Not implemented Type", bsdf_type)

    # Not implemented
    if bsdf_type == "thindielectric":
        bsdf_type = "dielectric"
    elif bsdf_type == "plastic":
        bsdf_type = "plastic"#"disney"
    elif bsdf_type == "roughplastic":
        bsdf_type = "diffuse" #"disney"
    material.type = bsdf_type

    # load texture
    texture = bsdf.find('texture')
    if texture is not None:
        texture_type = texture.attrib["type"]
        if texture_type == "bitmap":
            diffuse_map = texture.find('string[@name="filename"]').attrib["value"]
            material.diffuse_map = diffuse_map
        elif texture_type == "checkerboard":
            color0 = str2floatarray(texture.find('rgb[@name="color0"]').attrib["value"])
            color1 = str2floatarray(texture.find('rgb[@name="color1"]').attrib["value"])
            uoffset = float(texture.find('float[@name="uoffset"]').attrib["value"])
            voffset = float(texture.find('float[@name="voffset"]').attrib["value"])
            uscale = float(texture.find('float[@name="uscale"]').attrib["value"])
            vscale = float(texture.find('float[@name="vscale"]').attrib["value"])
            UV_mat = np.array([[uscale, 0, uoffset],
                               [0, vscale, voffset],
                               [0, 0, 1],
                               ], dtype=np.float32)
            material.color0 = color0
            material.color1 = color1
            material.to_uv = UV_mat
        else:
            print("Other texture type", texture_type)
    return material
