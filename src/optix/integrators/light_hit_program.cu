#include <optixu/optixu_math_namespace.h>
#include "optix/common/prd_struct.h"
#include "optix/light/light_parameters.h"
#include "optix/common/helpers.h"


using namespace optix;

rtDeclareVariable(SurfaceInteraction, si, rtPayload, );
rtDeclareVariable(PerRayData_pathtrace_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(int, lightId, , );
rtDeclareVariable(int, materialId, , );

// rtDeclareVariable(float3,     emission_color, , );
rtBuffer<LightParameter> sysLightParameters;
rtBuffer<MaterialParameter> sysMaterialParameters;

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable( float3, shading_normal, attribute shading_normal, );
rtDeclareVariable( float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(int, hitTriIdx,  attribute hitTriIdx, );

rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(unsigned int,     sample_type, , );
rtDeclareVariable(unsigned int,     use_mis, , );

RT_PROGRAM void diffuseEmitter()
{
    // Transform normal object to world coordinate
	// const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	const float3 world_geometric_normal = normalize(rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );

    // Face forwarding normal (ffnormal dot ray_direction > 0)
    float3 ff_normal = faceforward( world_geometric_normal, -ray.direction, world_geometric_normal );

    // Material parameter
	MaterialParameter& mat = sysMaterialParameters[materialId];
    float3 normal = mat.isTwosided? ff_normal : world_geometric_normal;

    si.material_id = materialId;
    si.normal = normal;

    float3 wi = -ray.direction;
    optix::Onb onb( normal );
    float3 wi_local = to_local(onb, wi);

    si.wi = wi_local;
    si.p = ray.origin + t_hit * ray.direction;

    // light specific codes
    si.light_id = lightId;
    si.hitTriIdx = hitTriIdx;
    LightParameter& light = sysLightParameters[lightId];
    const float3& emission_color = light.emission;
    si.emission = si.wi.z >= 1e-8? emission_color : make_float3(0.f);
    si.t = t_hit;
}

RT_PROGRAM void any_hit()
{
	prd_shadow.inShadow = true;
	rtTerminateRay();
}