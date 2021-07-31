#include "optix/integrators/common.h"
#include "optix/sampling/mis_guided_sampling.h"

#define MAX_NUM_VERTICES 32

using namespace optix;

namespace guided_path{
RT_FUNCTION void path_trace(Ray& ray, unsigned int& seed, PerPathData &ppd)
{

    float3 ray_origins[MAX_NUM_VERTICES];
    float3 ray_directions[MAX_NUM_VERTICES];
    float3 radiances[MAX_NUM_VERTICES];
    float3 current_attenuations[MAX_NUM_VERTICES];
    float3 direct_radiances[MAX_NUM_VERTICES];

    float emission_weight = 1.0;
    float3 throughput = make_float3(1.0);
    float3 result = make_float3(0.0);
    BSDFSample3f bs;

    // ---------------------- First intersection ----------------------
    SurfaceInteraction si;
    si.emission = make_float3(0);
    si.is_valid = true;
    rtTrace(top_object, ray, si);
    int depth;

#if USE_NEXT_EVENT_ESTIMATION
    int num_lights = sysLightParameters.size();
    float num_lights_inv = 1.0 / (float) (num_lights);
    LightSample lightSample;
    PerRayData_pathtrace_shadow prd_shadow;
#endif

    for (depth = 1; ; depth++){
        radiances[depth] = si.emission;
        ray_origins[depth] = si.p;

        // ---------------- Intersection with emitters ----------------
        result += emission_weight * throughput * si.emission;

        // ---------------- Terminate ray tracing ----------------
        // (1) over max depth
        // (2) ray missed
        // (3) hit emission
        if(depth >= max_depth || ! si.is_valid || (si.emission.x + si.emission.y + si.emission.z) > 0){
            break;
        }
        // (4) Russian roulette termination
        if(depth >= rr_begin_depth)
        {
            float pcont = fmaxf(throughput);
            pcont = max(pcont, 0.05);
            if(rnd(seed) >= pcont)
                break;
            throughput /= pcont;
        }

        MaterialParameter &mat = sysMaterialParameters[si.material_id];
        optix::Onb onb(si.normal);
#if USE_NEXT_EVENT_ESTIMATION
        // --------------------- Emitter sampling ---------------------
        float3 L = make_float3(0.0f);

        // sample light index
        int index = (num_lights==1)?0:optix::clamp(static_cast<int>(floorf(rnd(seed) * num_lights)), 0, num_lights - 1);
        const LightParameter& light = sysLightParameters[index];

        // sample light
        sample_light(si.p, light, seed, lightSample);

        float lightDist = lightSample.lightDist;
        float3 wo = lightSample.wi;
        float3 Li = lightSample.Li;
        float lightPdf = lightSample.pdf * num_lights_inv;
        //return make_float3(1.0f);
        bool is_light_delta = (light.lightType == LIGHT_POINT) || (light.lightType == LIGHT_DIRECTIONAL) || (light.lightType == LIGHT_SPOT);
        if ((!is_light_delta && (dot(wo, si.normal) <= 0.0f) )|| length(Li) == 0)
            L = make_float3(0.0);

        float3 wo_local = to_local(onb, wo);

        // Check visibility
        prd_shadow.inShadow = false;
        prd_shadow.seed = seed;
        optix::Ray shadowRay = optix::make_Ray(si.p, wo, 1, scene_epsilon, lightDist - scene_epsilon);
        rtTrace(top_shadower, shadowRay, prd_shadow);
        seed = prd_shadow.seed;

        if (!prd_shadow.inShadow)
        {
            float scatterPdf = mis_guided_sampling::Pdf(mat, si, onb, wo);
            float3 f = bsdf::Eval(mat, si, wo_local);

            // Delta light
            if(is_light_delta){
                L = Li * f / lightPdf;
            } else {
                float weight = powerHeuristic(lightPdf, scatterPdf);    // MIS
                L =  weight * Li * f / lightPdf;
            }
            // L *= float(num_lights);
        }
        direct_radiances[depth] = L;
        result += throughput * L;
#endif
        // --------------------- Surface scatter sampling ---------------------
        mis_guided_sampling::Sample(mat, si, onb, seed, bs);

        ray.origin = si.p;
        ray.direction = bs.wo;
        throughput *= bs.weight;

        // stop if throughput is zero
        if(dot(throughput, throughput) == 0){
            break;
        }

        // clear emission & trace again
        si.emission = make_float3(0.0);
        si.seed = seed;
        rtTrace(top_object, ray, si);
        seed = si.seed;

        ray_directions[depth] = ray.direction;
        current_attenuations[depth] = bs.weight;

#if USE_NEXT_EVENT_ESTIMATION
        /* Determine probability of having sampled that same
        direction using emitter sampling. */
        if(si.emission.x > 0)
        {
            LightParameter& light = sysLightParameters[si.light_id];
            float lightPdfArea = pdf_light(si.hitTriIdx, ray.origin, ray.direction, light);
            float light_pdf = (si.t * si.t) / si.wi.z * lightPdfArea * num_lights_inv;
            emission_weight = powerHeuristic(bs.pdf, light_pdf);
        }
#endif
    }

    float3 accumulated_radiance = radiances[depth];
    for(int i = depth-1; i>=1; i--){
        float q = (accumulated_radiance.x + accumulated_radiance.y + accumulated_radiance.z) / 3.0;
        accumulateQValue(ray_origins[i], ray_directions[i], q);
        if(i == depth-1){
            accumulated_radiance *= emission_weight;
        }
        accumulated_radiance = direct_radiances[i] + current_attenuations[i] * accumulated_radiance;
    }

    ppd.result = result;
    ppd.depth = depth;
    ppd.is_valid = si.is_valid;
}
}

