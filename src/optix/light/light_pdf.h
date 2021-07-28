#include "optix/light/light_parameters.h"
#include "optix/common/rt_function.h"
#include "optix/common/helpers.h"

RT_CALLABLE_PROGRAM float pdf_light_point(const float3 &pos, const float3 &wi, const LightParameter &light)
{
    return 0.0f;
}

RT_CALLABLE_PROGRAM float pdf_light_direction(const float3 &pos, const float3 &wi, const LightParameter &light)
{
    return 0.0f;
}

RT_CALLABLE_PROGRAM float pdf_light_spot(const float3 &pos, const float3 &wi, const LightParameter &light)
{
    return 0.0f;
}


RT_CALLABLE_PROGRAM float pdf_light_sphere(const float3 &pos, const float3 &wi, const LightParameter &light)
{
    float3 L = pos - light.position;
    float Ldist2 = dot(L, L);
    float Ldist = sqrtf(Ldist2);
    bool inside = Ldist <= light.radius;
    if(inside)
        return light.inv_area;
    float cos_theta_max = light.radius / Ldist;
    float area = (2 * M_PIf * (1 - cos_theta_max)) * light.radius * light.radius;
    return 1 / area;
}

RT_CALLABLE_PROGRAM float pdf_light_tri_mesh(const int hitTriIdx, const float3 &pos, const float3 &wi, const LightParameter &light)
{
    const int3 v_idx = light.indices_buffer_id[hitTriIdx];

    float3 p0 = light.pos_buffer_id[ v_idx.x ];
    float3 p1 = light.pos_buffer_id[ v_idx.y ];
    float3 p2 = light.pos_buffer_id[ v_idx.z ];

    p0 = transform_point(light.transformation, p0);
    p1 = transform_point(light.transformation, p1);
    p2 = transform_point(light.transformation, p2);

	float area = 0.5 * optix::length(optix::cross(p1- p0, p2 - p0));
	return 1 / area * ( 1.0 / light.n_triangles );
}

RT_CALLABLE_PROGRAM float pdf_light(const int hitTriIdx, const float3 &pos, const float3 &wi, const LightParameter &light)
{
    switch(light.lightType){
        case LIGHT_QUAD: return light.inv_area;
        case LIGHT_DISK: return light.inv_area;
        case LIGHT_SPHERE: return pdf_light_sphere(pos, wi, light);
//        case LIGHT_SPOT: return 0.0;
//        case LIGHT_POINT: return 0.0;
//        case LIGHT_DIRECTIONAL: return 0.0;
        case LIGHT_TRIANGLE_MESH: return pdf_light_tri_mesh(hitTriIdx, pos, wi, light);
    }
    return 0.0;
}