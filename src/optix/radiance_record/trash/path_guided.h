#include <optixu/optixu_math_namespace.h>

#include "optix/app_config.h"
#include "optix/light/light_sample.h"
#include "optix/light/light_pdf.h"
#include "optix/integrators/scatter_sampler.h"

#define MAX_NUM_VERTICES 32

using namespace optix;

// scene geometry + material + lights
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(rtObject,      top_shadower, , );
rtBuffer<MaterialParameter> sysMaterialParameters;
rtBuffer<LightParameter> sysLightParameters;

// path tracer
rtDeclareVariable(float,         scene_epsilon, , );

rtDeclareVariable(unsigned int,  rr_begin_depth, , );
rtDeclareVariable(unsigned int,  max_depth, , );

rtDeclareVariable(unsigned int,  pathtrace_ray_type, , );
rtDeclareVariable(unsigned int,  pathtrace_shadow_ray_type, , );


RT_FUNCTION float3 path_trace(Ray& ray, unsigned int& seed)
{

    float3 ray_origins[MAX_NUM_VERTICES];
    float3 ray_directions[MAX_NUM_VERTICES];
    float3 radiances[MAX_NUM_VERTICES];
    float3 current_attenuations[MAX_NUM_VERTICES];

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
            float pcont = min(fmaxf(throughput), 0.95f);
            if(rnd(seed) < pcont)
                break;
            throughput /= pcont;
        }

        MaterialParameter &mat = sysMaterialParameters[si.material_id];
        optix::Onb onb(si.normal);

        // --------------------- Surface scatter sampling ---------------------
        scatter::sample(mat, si, onb, seed, bs);
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
    }

    float3 accumulated_radiance = radiances[depth];
    for(int i = depth-1; i>=1; i--){
        float q = (accumulated_radiance.x + accumulated_radiance.y + accumulated_radiance.z) / 3.0;
        accumulateQValue(ray_origins[i], ray_directions[i], q);
        accumulated_radiance = radiances[i] + current_attenuations[i] * accumulated_radiance;
    }

    return result;
}
