#include "optix/cameras/perspective.h"
#include "optix/cameras/spherical.h"

using namespace optix;
rtDeclareVariable(unsigned int, camera_type, , );

RT_FUNCTION void generate_ray(float2 &pixel, float3 &origin, float3 &direction, unsigned int &seed)
{
    if(camera_type == 0){
        generate_ray_perspective(pixel, origin, direction, seed);
    } else if (camera_type == 1){
        generate_ray_spherical(pixel, origin, direction, seed);
    }
}