#include <optixu/optixu_math_namespace.h>
using namespace optix;
#include "optix/light/light_sample.h"

rtBuffer<LightParameter> lights;
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(rtObject, top_shadower, , );

rtDeclareVariable(float, sigma_s, , );
rtDeclareVariable(float, sigma_a, , );
rtDeclareVariable(float, sigma_t, , );
rtDeclareVariable(float, hg_g, , );

//const float sigma_s = 0.2;
//const float sigma_a = 0.0000;
//const float sigma_t = 0.2;
//const float hg_g = 0.9;



RT_CALLABLE_PROGRAM float3 DirectLight(MaterialParameter &mat,
    float3 &ffnormal, float3 &hit_point, float3 &wo, unsigned int &seed)
{
    float3 L = make_float3(0.0f);

    int num_lights = lights.size();

	//Pick a light to sample
	int index = optix::clamp(static_cast<int>(floorf(rnd(seed) * num_lights)), 0, num_lights - 1);
	LightParameter light = lights[index];
    LightSample lightSample;
    sample_light(hit_point, light, seed, lightSample);

	float3 surfacePos = hit_point;
	float3 surfaceNormal = ffnormal;

	float lightDist = lightSample.lightDist;
	float3 wi = lightSample.wi;
    float3 Li = lightSample.Li;
    float lightPdf = lightSample.pdf;
    //return make_float3(1.0f);
	if (dot(wi, surfaceNormal) <= 0.0f || length(Li) == 0)
		return L;

    // Check visibility
	PerRayData_pathtrace_shadow prd_shadow;
	prd_shadow.inShadow = false;
	optix::Ray shadowRay = optix::make_Ray(surfacePos, wi, 1, scene_epsilon, lightDist - scene_epsilon);
	rtTrace(top_object, shadowRay, prd_shadow);

	if (!prd_shadow.inShadow)
	{
	    float scatterPdf = Pdf(mat, ffnormal, wo, wi);
		float3 f = Eval(mat, ffnormal, wo, wi);

	    // Delta light
	    if(light.lightType == LIGHT_POINT || light.lightType == LIGHT_DIRECTIONAL){
	        L = Li * f / lightPdf;
	    } else {
	        float weight = powerHeuristic(lightPdf, scatterPdf);    // MIS
		    L =  weight * Li * f / lightPdf;
	    }
		L *= float(num_lights);
	}
	return L;
}

RT_CALLABLE_PROGRAM float3 DirectLightVolume(float3 &hit_point, float3 &scatter_out_dir, unsigned int &seed)
{
    float3 L = make_float3(0.0f);
    int num_lights = lights.size();

	//Pick a light to sample
	int index = optix::clamp(static_cast<int>(floorf(rnd(seed) * num_lights)), 0, num_lights - 1);
	LightParameter light = lights[index];
    LightSample lightSample;
    sample_light(hit_point, light, seed, lightSample);

	float3 surfacePos = hit_point;

	float lightDist = lightSample.lightDist;
	float3 wi = lightSample.wi;
    float3 Li = lightSample.Li;
    float lightPdf = lightSample.pdf;

    if (length(Li) == 0)
		return L;

    // Check visibility
	PerRayData_pathtrace_shadow prd_shadow;
	prd_shadow.inShadow = false;
	optix::Ray shadowRay = optix::make_Ray(surfacePos, wi, 1, scene_epsilon, lightDist - scene_epsilon);
	rtTrace(top_object, shadowRay, prd_shadow);


    if (!prd_shadow.inShadow)
	{
	    float3 scatter_in_dir = -wi;
	    float scatterPdf = HG_phase_function(hg_g, dot(scatter_out_dir, scatter_in_dir)) * (1/(4*M_PIf));
        float f = scatterPdf;

	    // Delta light
	    if(light.lightType == LIGHT_POINT || light.lightType == LIGHT_DIRECTIONAL){
	        L = Li * f / lightPdf;
	    } else {
		    float weight = powerHeuristic(lightPdf, scatterPdf);    // MIS
		    L = weight * Li * f / lightPdf;
	    }
	    // Attenuation
	    L *= exp(-sigma_t * lightDist);

		L *= float(num_lights);
	}
	return L;

//	if (!prd_shadow.inShadow)
//	{
//	    const float A = light.area;
//        float3 scatter_in_dir = -wi;
//		float NdotL = dot(lightSample.normal, -wi);
//		float lightPdf = lightDistSq / (A * NdotL);
//		// float lightPdf = prd.origin.x / (A * 4);
//
//		// float scatterPdf = Pdf(ffnormal, wo, wi);
//		// float f = HG_phase_function(hg_g, dot(wo, wi));
//
//		L = powerHeuristic(lightPdf, scatterPdf) * light.emission / max(0.001f, lightPdf) * exp(-sigma_t * lightDist) * scatterPdf;
//		L *= float(num_lights);
//		// L = make_float3(0.1f);
//	}
//	return L;
}