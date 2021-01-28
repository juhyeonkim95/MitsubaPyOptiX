#include "optix/common/rt_function.h"
using namespace optix;
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );

RT_FUNCTION void generate_ray_perspective(float2 &pixel, PerRayData_pathtrace& prd)
{
    // TODO : implement orthographic camera.

}