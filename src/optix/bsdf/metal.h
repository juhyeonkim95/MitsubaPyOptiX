#include <optixu/optixu_math_namespace.h>
#include "../integrators/optixPathTracer.h"
#include "../integrators/random.h"
#include "../integrators/helpers.h"
#include "../integrators/prd_struct.h"
#include "../integrators/qTable.cuh"

using namespace optix;

rtDeclareVariable(float3,     geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3,     shading_normal,   attribute shading_normal, );
rtDeclareVariable(float3, texcoord, attribute texcoord, );

rtDeclareVariable(optix::Ray, ray,              rtCurrentRay, );
rtDeclareVariable(float,      t_hit,            rtIntersectionDistance, );
rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );


rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );

RT_CALLABLE_PROGRAM float Pdf_conductor(float3 &ffnormal, float3 &wi)
{
    return 0.0f;
}

RT_CALLABLE_PROGRAM float3 Eval_conductor(float3 &ffnormal, float3 &wi)
{
    return make_float3(0.0f);
}

RT_CALLABLE_PROGRAM float3 Sample_conductor(float3 &ffnormal, float3 &wi)
{
    return make_float3(0.0f);
}

RT_PROGRAM void metal()
{
    float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
    float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
    float3 hitpoint = ray.origin + t_hit * ray.direction;
    //
    // Generate a reflection ray.  This will be traced back in ray-gen.
    //
    current_prd.origin = hitpoint;
    current_prd.direction = optix::reflect(ray.direction, ffnormal);
    current_prd.current_attenuation = make_float3(0.8, 0.6, 0.2);
    current_prd.t = t_hit;
}