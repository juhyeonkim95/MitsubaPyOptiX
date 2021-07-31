/*
 * Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <optixu/optixu_math_namespace.h>


#include "optix/common/prd_struct.h"
#include "optix/common/helpers.h"
#include "optix/common/rt_function.h"
#include "optix/cameras/camera.h"
#include "optix/app_config.h"

//#if SAMPLING_STRATEGY == SAMPLING_STRATEGY_BSDF
//#include "optix/integrators/path.h"
//#elif SAMPLING_STRATEGY == SAMPLING_STRATEGY_SD_TREE
#include "optix/integrators/path_guided_nee.h"
//#endif


using namespace optix;

// Scene wide variables

rtDeclareVariable(uint2,         launch_index, rtLaunchIndex, );

//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------


rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(unsigned int,  completed_sample_number, , );
rtDeclareVariable(unsigned int,  samples_per_pass, , );


rtBuffer<float4, 2>              output_buffer;
rtBuffer<float4, 2>              output_buffer2;

rtBuffer<unsigned int, 2>               hit_count_buffer;
rtBuffer<unsigned int, 2>               path_length_buffer;
rtBuffer<float2, 2>               scatter_type_buffer;

//rtDeclareVariable(unsigned int,  scatter_sample_type, , );
rtDeclareVariable(unsigned int,  need_q_table_update, , );
//rtDeclareVariable(unsigned int,     use_mis, , );
//rtDeclareVariable(unsigned int,     use_soft_q_update, , );
//rtDeclareVariable(unsigned int,     construct_stree, , );
//rtBuffer<float3, 3>              point_buffer;



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

        // Each iteration is a segment of the ray path.  The closest hit will
        // return new segments to be traced here.
        PerPathData ppd;
        path_trace(ray, seed, ppd);

        result += ppd.result;
        hit_count += dot(ppd.result, ppd.result) > 0 ? 1 : 0;
        // seed = prd.seed;
        // float hit_count = (prd.done && !prd.isMissed) ? 1.0 : 0.0;
        // float hit_count = (prd.done && (length(prd.result) > 0))  ? 1.0 : 0.0;
    } while (--left_samples_pass);

    output_buffer[launch_index] += make_float4(result, 1.0);
    hit_count_buffer[launch_index] += hit_count;
}


//-----------------------------------------------------------------------------
//
//  Exception program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void exception()
{
    //int2 screen_index = make_uint2(launch_index.x, launch_index.y);
    output_buffer[launch_index] = make_float4(bad_color, 1.0f);
}