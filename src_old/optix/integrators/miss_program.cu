#include <optixu/optixu_math_namespace.h>
#include "optix/common/prd_struct.h"
#include "optix/light/light_parameters.h"


using namespace optix;

rtDeclareVariable(PerRayData_pathtrace, prd, rtPayload, );
rtDeclareVariable(float3, bg_color, , );
rtBuffer<LightParameter> lights;


RT_PROGRAM void miss()
{
    prd.radiance = bg_color;
    prd.done = true;
    prd.t = 1000;
    prd.isMissed = true;
}

//RT_CALLABLE_PROGRAM float3 get_transformed_buffer(float3 v) {
//
//}

RT_PROGRAM void miss_environment_mapping()
{
    LightParameter light = lights[0];

    float3 ray = prd.direction;

    float4 a = make_float4(ray.x, ray.y, ray.z, 0);
    a = light.transformation.transpose() * a;
    ray = normalize(make_float3(a.x, a.y, a.z));

    float phi = atan2f(ray.x, -ray.z);
    float theta = acosf(-ray.y);
    float u = (phi + M_PIf) * (0.5f * M_1_PIf);
    float v = theta * M_1_PIf;


//    float theta = atan2f( ray.x, ray.z );
//    float phi   = M_PIf * 0.5f -  acosf( ray.y );
//    float u     = (theta + M_PIf) * (0.5f * M_1_PIf);
//    float v     = 0.5f * ( 1.0f + sin(phi) );

    const float3 emission = make_float3(optix::rtTex2D<float4>(light.envmapID , u, v));
    prd.radiance = emission;
    prd.done = true;
    prd.t = 1000;
    prd.isMissed = false;
}