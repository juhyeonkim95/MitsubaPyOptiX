#include <optixu/optixu_math_namespace.h>
#include "optix/raycasting/raycast_construct.h"
#include "optix/common/prd_struct.h"
#include "optix/bsdf/bsdf.h"

using namespace optix;

rtDeclareVariable( float3, shading_normal, attribute shading_normal, );
rtDeclareVariable( float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable( float3, texcoord, attribute texcoord, );

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

rtDeclareVariable(float3,     diffuse_color, , );
rtDeclareVariable(int,     diffuse_map_id, , );
rtDeclareVariable( Hit, hit_prd, rtPayload, );


RT_PROGRAM void closest_hit()
{
	const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	const float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
    float3 wo = -ray.direction;
    float3 hit_point = ray.origin + t_hit * ray.direction;
    float3 color = diffuse_color;

    MaterialParameter mat; // = sysMaterialParameters[materialId];
    mat.diffuse_color = diffuse_color;

    // BRDF Sampling
    float3 wi = Sample(mat, ffnormal, wo, hit_prd.seed);
    float pdf = Pdf(mat, ffnormal, wo, wi);
    float3 f = Eval(mat, ffnormal, wo, wi);

	hit_prd.t = t_hit;
	hit_prd.hit_point = hit_point;
	hit_prd.color = color;
	hit_prd.geom_normal = world_geometric_normal;
	hit_prd.new_direction = wi;
	hit_prd.pdf = pdf;
	hit_prd.attenuation *= f / pdf;
}