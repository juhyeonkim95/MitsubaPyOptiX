#include "optix/light/light_parameters.h"
#include "optix/common/rt_function.h"
#include "optix/common/helpers.h"

RT_CALLABLE_PROGRAM float pdf_light_point(float3 &pos, float3 &wi, LightParameter &light)
{
    return 0.0f;
}

RT_CALLABLE_PROGRAM float pdf_light_direction(float3 &pos, float3 &wi, LightParameter &light)
{
    return 0.0f;
}

RT_CALLABLE_PROGRAM float pdf_light_spot(float3 &pos, float3 &wi, LightParameter &light)
{
    return 0.0f;
}


RT_CALLABLE_PROGRAM float pdf_light_area(float3 &pos, float3 &wi, LightParameter &light)
{
    if (light.lightType == LIGHT_QUAD || light.lightType == LIGHT_DISK){
        return 1 / light.area;
    } else if (light.lightType == LIGHT_SPHERE) {
    	float3 L = pos - light.position;
    	float Ldist2 = dot(L, L);
    	float Ldist = sqrtf(Ldist2);
	    bool inside = Ldist <= light.radius;
	    if(inside)
            return 1 / light.area;
        float cos_theta_max = light.radius / Ldist;
        float area = (2 * M_PIf * (1 - cos_theta_max)) * light.radius * light.radius;
        return 1 / area;
    }
    return 0.0;
}

RT_CALLABLE_PROGRAM float pdf_light_tri_mesh(int hitTriIdx, float3 &pos, float3 &wi, LightParameter &light)
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

RT_CALLABLE_PROGRAM float pdf_light(int hitTriIdx, float3 &pos, float3 &wi, LightParameter &light)
{
    if (light.lightType == LIGHT_POINT){
        return pdf_light_point(pos, wi, light);
    } else if (light.lightType == LIGHT_DIRECTIONAL){
        return pdf_light_direction(pos, wi, light);
    } else if (light.lightType == LIGHT_QUAD || light.lightType == LIGHT_SPHERE || light.lightType == LIGHT_DISK){
        return pdf_light_area(pos, wi, light);
    } else if (light.lightType == LIGHT_SPOT){
        return pdf_light_spot(pos, wi, light);
    }
    else if (light.lightType == LIGHT_TRIANGLE_MESH){
        return pdf_light_tri_mesh(hitTriIdx, pos, wi, light);
    }
    return 0.0;
}