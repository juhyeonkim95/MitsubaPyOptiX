#include <optixu/optixu_math_namespace.h>
using namespace optix;
#include "optix/light/light_sample.h"
#include "optix/bsdf/bsdf.h"

rtBuffer<LightParameter> sysLightParameters;
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(rtObject, top_shadower, , );

rtDeclareVariable(float, sigma_s, , );
rtDeclareVariable(float, sigma_a, , );
rtDeclareVariable(float, sigma_t, , );
rtDeclareVariable(float, hg_g, , );


RT_CALLABLE_PROGRAM float3 DirectLight(
    const MaterialParameter &mat, const SurfaceInteraction &si, unsigned int &seed)
{
    float3 L = make_float3(0.0f);

    int num_lights = sysLightParameters.size();

	//Pick a light to sample
	int index = optix::clamp(static_cast<int>(floorf(rnd(seed) * num_lights)), 0, num_lights - 1);
	LightParameter light = sysLightParameters[index];
    LightSample lightSample;
    sample_light(si.p, light, seed, lightSample);

	float3 surfacePos = si.p;
	float3 surfaceNormal = si.normal;

	float lightDist = lightSample.lightDist;
	float3 wo = lightSample.wi;
    float3 Li = lightSample.Li;
    float lightPdf = lightSample.pdf;
    //return make_float3(1.0f);
    bool is_light_delta = (light.lightType == LIGHT_POINT) || (light.lightType == LIGHT_DIRECTIONAL) || (light.lightType == LIGHT_SPOT);
	if ((!is_light_delta && (dot(wo, surfaceNormal) <= 0.0f) )|| length(Li) == 0)
		return L;

    optix::Onb onb( si.normal );
    float3 wo_local = to_local(onb, wo);

    // Check visibility
	PerRayData_pathtrace_shadow prd_shadow;
	prd_shadow.inShadow = false;
	prd_shadow.seed = seed;
	optix::Ray shadowRay = optix::make_Ray(surfacePos, wo, 1, scene_epsilon, lightDist - scene_epsilon);
	rtTrace(top_shadower, shadowRay, prd_shadow);
    seed = prd_shadow.seed;

	if (!prd_shadow.inShadow)
	{
	    float scatterPdf = bsdf::Pdf(mat, si, wo_local);
		float3 f = bsdf::Eval(mat, si, wo_local);

	    // Delta light
	    if(is_light_delta){
	        L = Li * f / lightPdf;
	    } else {
	        float weight = powerHeuristic(lightPdf, scatterPdf);    // MIS
		    L =  weight * Li * f / lightPdf;
	    }
		L *= float(num_lights);
	}
	return L;
}