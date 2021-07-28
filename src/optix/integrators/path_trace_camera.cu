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
//#include "optix/light/direct_light.h"
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

rtDeclareVariable(uint3,         launch_index, rtLaunchIndex, );

//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------

//rtDeclareVariable(float3,        eye, , );
//rtDeclareVariable(float3,        U, , );
//rtDeclareVariable(float3,        V, , );
//rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(unsigned int,  frame_number, , );
rtDeclareVariable(unsigned int,  sqrt_num_samples, , );
rtDeclareVariable(unsigned int,  samples_per_pass, , );
rtDeclareVariable(unsigned int,  accumulative_q_table_update , , );


rtBuffer<float4, 2>              output_buffer;
rtBuffer<float4, 2>              output_buffer2;

rtBuffer<float, 2>               hit_count_buffer;
rtBuffer<float, 2>               path_length_buffer;
rtBuffer<float2, 2>               scatter_type_buffer;

rtDeclareVariable(unsigned int,  scatter_sample_type, , );
rtDeclareVariable(unsigned int,  need_q_table_update, , );
rtDeclareVariable(unsigned int,     use_mis, , );
rtDeclareVariable(unsigned int,     use_soft_q_update, , );
rtDeclareVariable(unsigned int,     construct_stree, , );

rtBuffer<float3, 3>              point_buffer;



RT_PROGRAM void pathtrace_camera()
{
    size_t2 screen = output_buffer.size();
    float2 inv_screen = 1.0f/make_float2(screen) * 2.f;
    uint2 screen_index = make_uint2(launch_index.x, launch_index.y);
    uint sample_index = launch_index.z;

    float2 pixel = make_float2(screen_index) * inv_screen - 1.f;

    float2 jitter_scale = inv_screen / sqrt_num_samples;
    // unsigned int samples_per_pixel = sqrt_num_samples*sqrt_num_samples;
    unsigned int left_samples_pass = samples_per_pass;
    float3 result = make_float3(0.0f);
    unsigned int completed_frame_number = frame_number;

    do
    {
        unsigned int seed = tea<16>(screen.x*screen.y*sample_index + screen.x*launch_index.y+launch_index.x, completed_frame_number);
        completed_frame_number += 1;

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
        float3 radiance = path_trace(ray, seed);

        result += radiance;
        // seed = prd.seed;
        // float hit_count = (prd.done && !prd.isMissed) ? 1.0 : 0.0;
        // float hit_count = (prd.done && (length(prd.result) > 0))  ? 1.0 : 0.0;

        // atomicAdd(&hit_count_buffer[screen_index], hit_count);
        // atomicAdd(&path_length_buffer[screen_index], float(prd.depth));
        // atomicAdd(&scatter_type_buffer[screen_index].x, prd.valid_scatter_count);
        // atomicAdd(&scatter_type_buffer[screen_index].y, prd.invalid_scatter_count);
    } while (--left_samples_pass);

    //prd.origin = ray_origin;
    //prd.direction = ray_direction;
    //prd.pdf = 0.0f;
    //prd.specularBounce = false;
    //
    // Update the output buffer
    //
//    result = result / (result + 1);
//    float3 old_color = make_float3(output_buffer[screen_index]);
//    output_buffer[screen_index] = make_float4( old_color * 0.99 + result * 0.01, 1.0f );

    atomicAdd(&output_buffer[screen_index].x,  result.x);
    atomicAdd(&output_buffer[screen_index].y,  result.y);
    atomicAdd(&output_buffer[screen_index].z,  result.z);
    atomicAdd(&output_buffer[screen_index].w,  samples_per_pass);

    //atomicAdd(&output_buffer2[screen_index].x,  result.x * result.x);
    //atomicAdd(&output_buffer2[screen_index].y,  result.y * result.y);
    //atomicAdd(&output_buffer2[screen_index].z,  result.z * result.z);
    //atomicAdd(&output_buffer2[screen_index].w,  samples_per_pass);

    //output_buffer[screen_index] = make_float4( result / samples_per_pass, 1.0 );
    //output_buffer2[screen_index] = make_float4( result * result / samples_per_pass, 1.0 );
}


//-----------------------------------------------------------------------------
//
//  Exception program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void exception()
{
    uint2 screen_index = make_uint2(launch_index.x, launch_index.y);
    output_buffer[screen_index] = make_float4(bad_color, 1.0f);
}