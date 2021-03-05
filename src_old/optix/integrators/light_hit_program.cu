#include <optixu/optixu_math_namespace.h>
#include "optix/common/prd_struct.h"
#include "optix/light/light_pdf.h"
#include "optix/common/helpers.h"
using namespace optix;

rtDeclareVariable(PerRayData_pathtrace, prd, rtPayload, );
rtDeclareVariable(PerRayData_pathtrace_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(int, lightId, , );

rtDeclareVariable(float3,     emission_color, , );
rtBuffer<LightParameter> lights;

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable( float3, shading_normal, attribute shading_normal, );
rtDeclareVariable( float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(int, hitTriIdx,  attribute hitTriIdx, );

rtDeclareVariable(float, hit_dist, rtIntersectionDistance, );
rtDeclareVariable(unsigned int,     sample_type, , );
rtDeclareVariable(unsigned int,     use_mis, , );

RT_PROGRAM void diffuseEmitter()
{
    const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	const float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
	float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
    LightParameter light = lights[lightId];

    float3 normal = world_shading_normal;
    if (light.isTwosided == 1){
	    normal = ffnormal;
	}

    float NdotL = dot(normal, -ray.direction);

    if(prd.depth == 0 || use_mis == 0){
        prd.radiance = NdotL >= 0? emission_color : make_float3(0.f);
    } else {

        float lightPdfArea = pdf_light(hitTriIdx, ray.origin, ray.direction, light);

        // float A = light.area;//length(cross(light.v1, light.v2));
        float lightPdf = (hit_dist * hit_dist) / clamp(NdotL, 1.e-3f, 1.0f) * lightPdfArea;

        float mis_weight = powerHeuristic(prd.scatterPdf, lightPdf);
        prd.radiance = NdotL >= 0 ? mis_weight * emission_color : make_float3(0.f);
    }
    prd.t = hit_dist;
    prd.done = true;
}

RT_PROGRAM void any_hit()
{
	prd_shadow.inShadow = true;
	rtTerminateRay();
}