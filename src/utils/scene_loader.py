import xml.etree.ElementTree as ET
import numpy as np
from core.camera import Camera
from core.parameters import MaterialParameter, ShapeParameter


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

    return camera

def load_scene_size(sensor):
    height = int(sensor.find('film/integer[@name="height"]').attrib['value'])
    width = int(sensor.find('film/integer[@name="width"]').attrib['value'])
    return height, width

def load_material(root):
    material_dictionary = {}
    texture_list = []

    for bsdf in root.findall('bsdf'):
        bsdf_type = bsdf.attrib['type']
        bsdf_id = bsdf.attrib['id']

        material = MaterialParameter(bsdf_id)
        if bsdf_type == "twosided":
            bsdf = bsdf.find("bsdf")
            bsdf_type = bsdf.attrib['type']

        if bsdf_type == "dielectric":
            material.intIOR = float(bsdf.find('float[@name="intIOR"]').attrib["value"])
            material.type = bsdf_type

        if bsdf_type == "diffuse":
            spectrum = bsdf.find("spectrum")
            rgb = bsdf.find("rgb")
            texture = bsdf.find('texture')
            if spectrum is not None:
                diffuse_color = str2floatarray(spectrum.attrib["value"])
                if diffuse_color.shape[0] == 1:
                    diffuse_color = np.tile(diffuse_color, 3)
                material.color = diffuse_color
            elif rgb is not None:
                diffuse_color = str2floatarray(rgb.attrib["value"])
                if diffuse_color.shape[0] == 1:
                    diffuse_color = np.tile(diffuse_color, 3)
                material.color = diffuse_color
            elif texture is not None:
                diffuse_map = texture.find('string[@name="filename"]').attrib["value"]

                if diffuse_map not in texture_list:
                    texture_list.append(diffuse_map)
                    #texture_dictionary[diffuse_map] = len(texture_dictionary)
                material.diffuse_map = diffuse_map

        material_dictionary[bsdf_id] = material
    return texture_list, material_dictionary


def load_shape(root, material_parameter_dict):
    shape_list = []
    obj_list = []
    for shape in root.findall('shape'):
        shape_type = shape.attrib['type']
        material_ref = shape.find("ref")
        if material_ref is None:
            continue
        material_id = material_ref.attrib["id"]
        shape_parameter = ShapeParameter()
        shape_parameter.shape_type = shape_type
        shape_parameter.material_parameter = material_parameter_dict[material_id]

        if shape_type == "rectangle":
            to_world = shape.find("transform/matrix").attrib['value']
            to_world = str2_4by4mat(to_world)
            shape_parameter.rectangle_info = rectangle(to_world)
        elif shape_type == "cube":
            to_world = shape.find("transform/matrix").attrib['value']
            to_world = str2_4by4mat(to_world)
            shape_parameter.rectangle_info = cube(to_world)
        elif shape_type == "sphere":
            radius = float(shape.find("float").attrib['value'])
            point_x = float(shape.find("point").attrib['x'])
            point_y = float(shape.find("point").attrib['y'])
            point_z = float(shape.find("point").attrib['z'])
            center = np.array([point_x, point_y, point_z], dtype=np.float32)
            shape_parameter.radius = radius
            shape_parameter.center = center

        elif shape_type == "obj":
            mesh_file_name = shape.find('string[@name="filename"]').attrib["value"]
            if mesh_file_name not in obj_list:
                obj_list.append(mesh_file_name)

            transformation = shape.find('transform[@name="toWorld"]/matrix').attrib["value"]
            shape_parameter.transformation = str2_4by4mat(transformation)
            shape_parameter.obj_file_name = mesh_file_name

        emitter = shape.find("emitter")
        if emitter is not None:
            spectrum_radiance = emitter.find('spectrum[@name="radiance"]')
            rgb_radiance = emitter.find('rgb[@name="radiance"]')

            if spectrum_radiance is not None:
                radiance = spectrum_radiance.attrib['value']
            if rgb_radiance is not None:
                radiance = rgb_radiance.attrib['value']
            radiance = str2floatarray(radiance)
            material_parameter = MaterialParameter("light_")
            material_parameter.type = 'light'
            material_parameter.emission = radiance
            shape_parameter.material_parameter = material_parameter
            #material_dictionary[shape_id] = {"type": "light", "radiance": radiance}
        shape_list.append(shape_parameter)

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

    return new_shape_list, obj_list


# def load_scene(file_name):
#     doc = ET.parse(file_name)
#     root = doc.getroot()
#
#     # 1. load camera
#     camera = load_camera(root.find("sensor"))
#
#     # 2. load material info
#     texture_id_dictionary, material_dictionary = load_material(root)
#
#     # 3. load shape info
#     shape_list, obj_list = load_shape(root)



    #
    # #
    # # material_dictionary = {}
    # # for bsdf in root.findall('bsdf'):
    # #     bsdf_id = bsdf.attrib['id']
    # #     for bsdf_attrib in bsdf.findall('bsdf'):
    # #         bsdf_attrib_name = bsdf_attrib.attrib['type']
    # #         diffuse = str2floatarray(bsdf_attrib.find('rgb').attrib['value'])
    # #         material_dictionary[bsdf_id] = {"type": "diffuse", "diffuse": diffuse}
    # #     for additional_attrib in bsdf.findall('float'):
    # #         if additional_attrib.attrib['name'] == "intIOR":
    # #             material_dictionary[bsdf_id] = {"type": "dielectric", "ior": float(additional_attrib.attrib['value'])}
    #
    # shape_list = []
    #
    # for shape in root.findall('shape'):
    #     shape_type = shape.attrib['type']
    #     material_ref = shape.find("ref")
    #     if material_ref is None:
    #         continue
    #     material_id = material_ref.attrib["id"]
    #
    #     if shape_type == "rectangle":
    #         to_world = shape.find("transform/matrix").attrib['value']
    #         to_world = str2_4by4mat(to_world)
    #         rectangle_info = rectangle(to_world)
    #         #if shape_id == "BackWall":
    #         o, u, v =rectangle_info
    #         cen = o + u * 0.5 + v * 0.5
    #         if shape_id == "Light":
    #             u = u*100
    #             v = v*100
    #             o = cen - 0.5*u - 0.5*v
    #         cen = o + u * 0.5 + v * 0.5
    #         print(shape_id, cen, np.cross(u, v))
    #         #print(shape_id, rectangle_info[0])
    #         shape_dictionary[shape_id] = {"type":"rectangle", "data":(o, u, v)}
    #     elif shape_type == "sphere":
    #         radius = float(shape.find("float").attrib['value'])
    #         point_x = float(shape.find("point").attrib['x'])
    #         point_y = float(shape.find("point").attrib['y'])
    #         point_z = float(shape.find("point").attrib['z'])
    #         center = np.array([point_x, point_y, point_z], dtype=np.float32)
    #         shape_dictionary[shape_id] = {"type": "sphere", "radius": radius, "center": center}
    #     elif shape_type == "obj":
    #         mesh_file_name = shape.find('./string[@name="filename"]').attrib["value"]
    #         transformation = shape.find('./transform[@name="toWorld"]/matrix').attrib["value"]
    #         transformation = str2_4by4mat(transformation)
    #
    #         mesh = CustomOptixMesh()
    #         mesh.load_from_file(mesh_file_name)
    #
    #     emitter = shape.find("emitter")
    #     if emitter is not None:
    #         radiance = emitter.find('rgb').attrib['value']
    #         radiance = str2floatarray(radiance)
    #         material_dictionary[shape_id] = {"type": "light", "radiance": radiance}
    #
    # print(material_dictionary)
    # print(shape_dictionary)
    # return camera_fov, camera_matrix, material_dictionary, shape_dictionary