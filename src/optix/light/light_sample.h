#include "optix/light/light_parameters.h"
#include "optix/common/helpers.h"
#include "optix/common/sampling.h"
#include "optix/common/rt_function.h"
#include "optix/common/random.h"

using namespace optix;



RT_FUNCTION void sample_light_area_sphere(const float3& p, const LightParameter &light, unsigned int &seed, AreaSample &sample)
{
	const float r1 = rnd(seed);
	const float r2 = rnd(seed);
	float3 L = p - light.position;
	bool inside = dot(L, L) <= light.radius * light.radius;
	if(inside){
	    float3 dir = UniformSampleSphere(r1, r2);
        sample.position = light.position + dir * light.radius;
        sample.normal = dir;
        sample.pdf = 1 / light.area;
        return;
	}
	float pdf;
	float3 dir = UniformSampleSphereGivenP(p, light.position, light.radius, r1, r2, pdf);
	sample.position = light.position + dir * light.radius;
	sample.normal = dir;
	sample.pdf = pdf;
}

RT_FUNCTION void sample_light_area_quad(const LightParameter &light, unsigned int &seed, AreaSample &sample)
{
	const float r1 = rnd(seed);
	const float r2 = rnd(seed);
	sample.position = light.position + light.u * r1 + light.v * r2;
	sample.normal = light.normal;
	sample.pdf = 1 / light.area;
}

RT_FUNCTION void sample_light_area_triangle_mesh(const LightParameter &light, unsigned int &seed, AreaSample &sample)
{
    int tri_index = optix::clamp(static_cast<int>(floorf(rnd(seed) * light.n_triangles)), 0, light.n_triangles - 1);
    const int3 v_idx = light.indices_buffer_id[tri_index];

    float3 p0 = light.pos_buffer_id[ v_idx.x ];
    float3 p1 = light.pos_buffer_id[ v_idx.y ];
    float3 p2 = light.pos_buffer_id[ v_idx.z ];

    p0 = transform_point(light.transformation, p0);
    p1 = transform_point(light.transformation, p1);
    p2 = transform_point(light.transformation, p2);

	const float r1 = rnd(seed);
	const float r2 = rnd(seed);
    float2 b = UniformSampleTriangle(r1, r2);
	sample.position = b.x * p0 + b.y * p1 + (1- b.x - b.y) * p2;

    if (light.normal_buffer_id != RT_BUFFER_ID_NULL){
        float3 n0 = light.normal_buffer_id[ v_idx.x ];
        float3 n1 = light.normal_buffer_id[ v_idx.y ];
        float3 n2 = light.normal_buffer_id[ v_idx.z ];
        float3 normal = b.x * n0 + b.y * n1 + (1- b.x - b.y) * n2;
        sample.normal = transform_normal(light.transformation, normal);
    } else {
        sample.normal = optix::normalize(optix::cross(p1- p0, p2 - p0));
    }
	float area = 0.5 * optix::length(optix::cross(p1- p0, p2 - p0));
	sample.pdf = 1 / area * (1.0 / light.n_triangles);
}

RT_FUNCTION void sample_light_area_disk(const LightParameter &light, unsigned int &seed, AreaSample &sample)
{
	const float2 u = make_float2(rnd(seed), rnd(seed));
	float2 r_theta = square_to_disk(u);
	float r = r_theta.x;
	float theta = r_theta.y;
	float3 dir = make_float3(cosf(theta), sinf(theta), 0);
	optix::Onb onb( light.normal );
    onb.inverse_transform( dir );
    sample.position = light.position + light.radius * r * dir;
	sample.normal = light.normal;
	sample.pdf = 1 / light.area;
}

RT_FUNCTION void sample_light_area(const float3 &pos, const LightParameter &light, unsigned int &seed, LightSample &sample)
{
    AreaSample areaSample;
    switch(light.lightType){
        case LIGHT_QUAD:sample_light_area_quad(light, seed, areaSample);break;
        case LIGHT_SPHERE:sample_light_area_sphere(pos, light, seed, areaSample);break;
        case LIGHT_DISK:sample_light_area_disk(light, seed, areaSample);break;
        case LIGHT_TRIANGLE_MESH:sample_light_area_triangle_mesh(light, seed, areaSample);break;
    }

    float3 lightDir = areaSample.position - pos;
    float lightDist = length(lightDir);
	float lightDistSq = lightDist * lightDist;
    lightDir /= lightDist;

    float lightAreaPdf = areaSample.pdf;
    float NdotL = dot(areaSample.normal, -lightDir);

    if(NdotL <= 0.0f){
        sample.Li = make_float3(0.0);
        sample.pdf = 1.0;
        return;
    }
    float lightPdf = lightDistSq / NdotL * lightAreaPdf;

    // direction is opposite
    sample.Li = light.emission;
	sample.wi = lightDir;
    sample.pdf = lightPdf;
    sample.lightDist = lightDist;
}

RT_CALLABLE_PROGRAM void sample_light_point(const float3 &pos, const LightParameter &light, unsigned int &seed, LightSample &sample)
{
    float3 lightDir = light.position - pos;
    float lightDist = length(lightDir);
    float lightDistSq = lightDist * lightDist;
    lightDir = normalize(lightDir);
	sample.Li = light.intensity / lightDistSq;
	sample.wi = lightDir;
    sample.pdf = 1.0f;
    sample.lightDist = lightDist;
}

RT_CALLABLE_PROGRAM void sample_light_spot(const float3 &pos, const LightParameter &light, unsigned int &seed, LightSample &sample)
{
    float3 lightDir = light.position - pos;
    float lightDist = length(lightDir);
    float lightDistSq = lightDist * lightDist;
    lightDir /= lightDist;

    float falloff;

    float cosTheta = dot(-light.direction, lightDir);
    float cosTotalWidth = light.cosTotalWidth;
    float cosFalloffStart = light.cosFalloffStart;

    if(cosTheta < cosTotalWidth){
        falloff = 0;
    } else if(cosTheta > cosFalloffStart){
        falloff = 1;
    } else {
        float delta = (cosTheta - cosTotalWidth) / (cosFalloffStart - cosTotalWidth);
        falloff = (delta * delta) * (delta * delta);
    }

	sample.Li = falloff * light.intensity / lightDistSq;
	sample.wi = lightDir;
    sample.pdf = 1.0f;
    sample.lightDist = lightDist;
}

RT_CALLABLE_PROGRAM void sample_light_direction(const float3 &pos, const LightParameter &light, unsigned int &seed, LightSample &sample)
{
    sample.Li = light.emission;
    sample.wi = light.direction;
    sample.pdf = 1.0f;
    sample.lightDist = 9999; // big number
}

RT_CALLABLE_PROGRAM void sample_light(const float3 &pos, const LightParameter &light, unsigned int &seed, LightSample &sample)
{
    if (light.lightType == LIGHT_QUAD ||
               light.lightType == LIGHT_SPHERE ||
               light.lightType == LIGHT_DISK ||
               light.lightType == LIGHT_TRIANGLE_MESH
    ){
        sample_light_area(pos, light, seed, sample);
    } else if (light.lightType == LIGHT_SPOT){
        sample_light_spot(pos, light, seed, sample);
    } else if (light.lightType == LIGHT_POINT){
        sample_light_point(pos, light, seed, sample);
    } else if (light.lightType == LIGHT_DIRECTIONAL){
        sample_light_direction(pos, light, seed, sample);
    }
}