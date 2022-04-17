#include <optixu/optixu_math_namespace.h>

#include "optix/common/prd_struct.h"
#include "optix/common/helpers.h"
#include "optix/common/rt_function.h"
#include "optix/cameras/camera.h"
#include "optix/app_config.h"
#include "optix/integrators/path.h"

using namespace optix;

// Scene wide variables
rtDeclareVariable(uint2,         launch_index, rtLaunchIndex, );

//-----------------------------------------------------------------------------
//  Camera program -- main ray tracing loop
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(unsigned int,  completed_sample_number, , );
rtDeclareVariable(unsigned int,  samples_per_pass, , );

rtBuffer<float4, 2>              output_buffer;
rtBuffer<unsigned int, 2>               hit_count_buffer;


RT_PROGRAM void pathtrace_camera()
{
    size_t2 screen = output_buffer.size();
    float2 inv_screen = 1.0f/make_float2(screen) * 2.f;

    float2 pixel = make_float2(launch_index) * inv_screen - 1.f;

    unsigned int left_samples_pass = samples_per_pass;
    float3 result = make_float3(0.0f);

    // should be larger than 0!
    unsigned int sample_index_offset = completed_sample_number + 1;
    unsigned int hit_count = 0;
    PerPathData ppd;

    do
    {
        unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, sample_index_offset);
        sample_index_offset += 1;

        // Independent jittering
        float2 jitter = make_float2(rnd(seed), rnd(seed));
        float2 d = pixel + jitter * inv_screen;

        // Generate initial ray
        float3 ray_origin;
        float3 ray_direction;
        generate_ray(d, ray_origin, ray_direction, seed);
        Ray ray = make_Ray(ray_origin, ray_direction, pathtrace_ray_type, scene_epsilon, RT_DEFAULT_MAX);

        // Perform path trace
        path::path_trace(ray, seed, ppd);
        result += ppd.result;
        hit_count += dot(ppd.result, ppd.result) > 0 ? 1 : 0;

    } while (--left_samples_pass);

    output_buffer[launch_index] += make_float4(result, 1.0);
    hit_count_buffer[launch_index] += hit_count;
}

//-----------------------------------------------------------------------------
//  Exception program
//-----------------------------------------------------------------------------

RT_PROGRAM void exception()
{
    output_buffer[launch_index] = make_float4(bad_color, 1.0f);
}
