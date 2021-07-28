#include "optix/common/rt_function.h"
#include "optix/common/helpers.h"
#include "optix/common/random.h"

using namespace optix;
rtDeclareVariable(float3,        spherical_cam_position, , );
rtDeclareVariable(float3,        spherical_cam_size, , );
rtDeclareVariable(unsigned int,        spherical_cam_directional_mapping, , );

RT_FUNCTION void generate_ray_spherical(const float2 &pixel, float3 &origin, float3 &direction, unsigned int &seed)
{
    float3 rayOrigin = spherical_cam_position + spherical_cam_size * make_float3(rnd(seed), rnd(seed), rnd(seed));

    float2 normalized_pixel = (pixel + 1) * 0.5;
    float3 rayDirection;
    switch(spherical_cam_directional_mapping){
        case 0: rayDirection = mapUVToDirectionSphere(normalized_pixel); break;
        case 1: rayDirection = mapCanonicalToDirection(normalized_pixel); break;
    }
    //float3 rayDirection = mapCanonicalToDirection((pixel + 1) * 0.5);
    //float3 rayDirection = mapUVToDirection(normalized_pixel);
    origin = rayOrigin;
    direction = rayDirection;
}
