#include <optixu/optixu_math_namespace.h>
#include "optix/raycasting/raycast_construct.h"


using namespace optix;

rtDeclareVariable( float3, shading_normal, attribute shading_normal, );
rtDeclareVariable( float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable( float3, texcoord, attribute texcoord, );

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(float3,     emission_color, , );

rtDeclareVariable( Hit, hit_prd, rtPayload, );


RT_PROGRAM void closest_hit()
{
	const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	const float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    const float3 hit_point = ray.origin + t_hit * ray.direction;

	hit_prd.t = t_hit;
	hit_prd.hit_point = hit_point;
	hit_prd.geom_normal = world_geometric_normal;
	hit_prd.color = emission_color;
	hit_prd.done = 1;
	hit_prd.result += emission_color * hit_prd.attenuation;
}
