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
//#include "optixPathTracer.h"
//#include "random.h"
//#include "helpers.h"
//#include "prd_struct.h"
#include "optix/q_table/qTable.cuh"
#include "optix/common/helpers.h"
#include "optix/light/direct_light.h"
#include "optix/common/rt_function.h"
#include "optix/cameras/camera.h"

using namespace optix;

// Scene wide variables
// rtDeclareVariable(float,         scene_epsilon, , );
// rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(uint3,         launch_index, rtLaunchIndex, );

//rtDeclareVariable(float,         sigma_s, , );
//rtDeclareVariable(float,         sigma_a, , );
//rtDeclareVariable(float,         sigma_t, , );
//rtDeclareVariable(float,         hg_g, , );


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
rtDeclareVariable(unsigned int,  rr_begin_depth, , );
rtDeclareVariable(unsigned int,  max_depth, , );
rtDeclareVariable(unsigned int,  pathtrace_ray_type, , );
rtDeclareVariable(unsigned int,  pathtrace_shadow_ray_type, , );
rtDeclareVariable(unsigned int,  samples_per_pass, , );
rtDeclareVariable(unsigned int,  accumulative_q_table_update , , );


rtBuffer<float4, 2>              output_buffer;
rtBuffer<float4, 2>              output_buffer2;
rtBuffer<float, 2>               hit_count_buffer;
rtBuffer<float, 2>               path_length_buffer;

rtDeclareVariable(unsigned int,  scatter_sample_type, , );
rtDeclareVariable(unsigned int,  need_q_table_update, , );
rtDeclareVariable(unsigned int,     use_mis, , );
rtDeclareVariable(unsigned int,     use_soft_q_update, , );


RT_CALLABLE_PROGRAM bool volumeScatter(float3 ray_origin, float3 ray_direction, PerRayData_pathtrace &prd)
{
    if(sigma_t > 0.0){
        float tout = prd.t;
        float tin = scene_epsilon;
        float prob_s =  1 - exp(-sigma_s * (tout - tin));
        float scaleBy = 1.0/(1.0-prob_s);

        // scatter occurs.
        if(rnd(prd.seed) < prob_s){
            float s = sampleSegment(rnd(prd.seed), sigma_s, tout - tin);
            float3 x = ray_origin + ray_direction * (tin + s);

            float3 dir;
            if(scatter_sample_type == 0){
                dir = sampleSphere(rnd(prd.seed), rnd(prd.seed));
                prd.attenuation *= HG_phase_function(hg_g, dot(dir, ray_direction));
                // float3 u, v;
                // generateOrthoBasis(u, v, ray_direction);
                // dir = u*dir.x+v*dir.y+ray_direction*dir.z;
                float pdf = HG_phase_function(hg_g, dot(dir, ray_direction)) * 1.0 / (4 * M_PIf);
                prd.scatterPdf = pdf;
            }else if(scatter_sample_type ==1){
                dir = sampleHG(hg_g, rnd(prd.seed), rnd(prd.seed));
                // optix::Onb onb( ray_direction );
                // onb.inverse_transform(dir);
                float3 u, v;
                generateOrthoBasis(u, v, ray_direction);
                dir = u*dir.x+v*dir.y+ray_direction*dir.z;
                float pdf = HG_phase_function(hg_g, dot(dir, ray_direction)) * 1.0 / (4 * M_PIf);
                prd.scatterPdf = pdf;
            }else if(scatter_sample_type == 2){
                Sample_info sample_info = sampleScatteringDirectionProportionalToQVolume(x, prd.seed);
                dir = sample_info.direction;
                float pdf = sample_info.pdf;
                float a = HG_phase_function(hg_g, dot(dir, ray_direction)) / (pdf);
                prd.attenuation *= a;
            }else if(scatter_sample_type == 3){
                Sample_info sample_info = sampleScatteringDirectionProportionalToQVolumeHG(x, ray_direction, hg_g, prd.seed);
                dir = sample_info.direction;
                float pdf = sample_info.pdf;
                float a = HG_phase_function(hg_g, dot(dir, ray_direction)) / (pdf);
                prd.attenuation *= a;
            }

            //optix::Onb onb( ray.direction );
            //onb.inverse_transform( dir );
            // ray_origin = x;
            // ray_direction = dir;
            prd.origin = x;
            prd.direction = dir;
            prd.done = false;
            prd.depth++;
            prd.volume_scattered = true;
            float3 scatter_out_dir = -ray_direction;
            if(use_mis==1)
                prd.radiance = DirectLightVolume(x, scatter_out_dir, prd.seed);

            // prd.current_attenuation = prob_s * (1.0/prob_s);
            return true;
        }
        // scatter doesn't occur.
        else {
            prd.attenuation *= exp(-sigma_t * tout);
            prd.attenuation *= scaleBy;
            return false;
            //prd.attenuation *= prd.current_attenuation;
        }
    }
    return false;
}


RT_FUNCTION void integrator(PerRayData_pathtrace& prd, float3& radiance)
{
    radiance = make_float3(0.0f);

    float3 ray_origin = prd.origin;
    float3 ray_direction = prd.direction;
    float3 normal = make_float3(0,1,0);

    for(;;)
    {
        prd.current_attenuation = make_float3(1.0f);
        prd.volume_scattered = false;
        prd.radiance = make_float3(0.0f);
        Ray ray = make_Ray(ray_origin, ray_direction, pathtrace_ray_type, scene_epsilon, RT_DEFAULT_MAX);
        rtTrace(top_object, ray, prd);
        bool scattered = false;
        if(!prd.isMissed){
            scattered = volumeScatter(ray_origin, ray_direction, prd);
        }
        if(need_q_table_update == 1){
            float reward = fmaxf(prd.radiance);
            float target_q_value;
            if(prd.done){
                target_q_value = reward * 0.01;
            }else{
                float3 wo = -ray_direction;
                float3 wi = prd.direction;

                if(prd.volume_scattered){
                    float distance = length(prd.origin - ray_origin);
                    target_q_value = getNextQValueVolume(hg_g, prd.origin, wo, wi) * exp(-sigma_t * distance);
                } else {
                    //new_value = reward + 0.9f * getQValue(prd.origin, prd.direction);
                    //float f_s = fmaxf(prd.mat.color);
                    float f_s = (prd.diffuse_color.x + prd.diffuse_color.y + prd.diffuse_color.z) / 3.0f;

                    target_q_value = getNextQValue(prd.origin, prd.normal, wo, wi) * f_s;
                    if(sigma_t > 0){
                        float distance = length(prd.origin - ray_origin);
                        target_q_value *= exp(-sigma_t * distance);
                    }
                }
            }

            if(prd.depth > 0){
                if(save_q_cos == 1){
                    target_q_value *= max(dot(normal, ray_direction), 0.0);
                }

                if(accumulative_q_table_update == 0){
                    float alpha = 1.0f / sqrt(1.0f + updateVisit(ray_origin, ray_direction));
                    float update_value = (1-alpha) * getQValue(ray_origin, ray_direction) + alpha * target_q_value;
                    setQValue(ray_origin, ray_direction, update_value);

                    if(use_soft_q_update)
                        setQValueSoft(ray_origin, prd.origin, target_q_value);
                } else if(accumulative_q_table_update == 1){
                    accumulateQValue(ray_origin, ray_direction, target_q_value);
                }
            }
        }

        if(!scattered){
            prd.result += prd.radiance * prd.attenuation;
            prd.attenuation *= prd.current_attenuation;
            if(prd.done)
            {
                // prd.result += prd.radiance * prd.attenuation;
                break;
            }
        }

        // Russian roulette termination
        if(prd.depth >= rr_begin_depth)
        {
            float pcont = fmaxf(prd.attenuation);
            pcont = max(pcont, 0.05);
            if(rnd(prd.seed) >= pcont)
                break;
            prd.attenuation /= pcont;
        }

        if(prd.depth >= max_depth){
            break;
        }

        prd.depth++;

        // Update ray data for the next path segment
        ray_origin = prd.origin;
        ray_direction = prd.direction;
        normal = prd.normal;
    }
}

RT_PROGRAM void pathtrace_camera()
{
    size_t2 screen = output_buffer.size();
    float2 inv_screen = 1.0f/make_float2(screen) * 2.f;
    uint2 screen_index = make_uint2(launch_index.x, launch_index.y);
    uint sample_index = launch_index.z;

    float2 pixel = make_float2(screen_index) * inv_screen - 1.f;

    float2 jitter_scale = inv_screen / sqrt_num_samples;
    unsigned int samples_per_pixel = sqrt_num_samples*sqrt_num_samples;
    unsigned int left_samples_pass = samples_per_pass;
    float3 result = make_float3(0.0f);
    unsigned int completed_frame_number = frame_number;

    do
    {
        unsigned int seed = tea<16>(screen.x*screen.y*sample_index + screen.x*launch_index.y+launch_index.x, completed_frame_number);
        completed_frame_number += 1;
        //
        // Sample pixel using jittering
        //
//        unsigned int x = samples_per_pixel%sqrt_num_samples;
//        unsigned int y = samples_per_pixel/sqrt_num_samples;
//        float2 jitter = make_float2(x-rnd(seed), y-rnd(seed));
//        float2 d = pixel + jitter*jitter_scale;

        // independent
        float2 jitter = make_float2(rnd(seed), rnd(seed));
        float2 d = pixel + jitter * inv_screen;

        // Initialze per-ray data
        PerRayData_pathtrace prd;
        prd.result = make_float3(0.f);
        prd.radiance = make_float3(0.f);
        prd.attenuation = make_float3(1.f);
        prd.countEmitted = true;
        prd.done = false;
        prd.isMissed = false;
        prd.seed = seed;
        prd.depth = 0;
        prd.normal = make_float3(0, 1, 0);
        generate_ray(d, prd.origin, prd.direction, prd.seed);

        // Each iteration is a segment of the ray path.  The closest hit will
        // return new segments to be traced here.
        float3 radiance;
        integrator(prd, radiance);

        result += prd.result;
        seed = prd.seed;
        float hit_count = (prd.done && !prd.isMissed) ? 1.0 : 0.0;
        //atomicAdd(&hit_count_buffer[screen_index], hit_count);
        //atomicAdd(&path_length_buffer[screen_index], float(prd.depth));
        hit_count_buffer[screen_index] += hit_count;
        path_length_buffer[screen_index] += float(prd.depth);

    } while (--left_samples_pass);

    //prd.origin = ray_origin;
    //prd.direction = ray_direction;
    //prd.pdf = 0.0f;
    //prd.specularBounce = false;
    //
    // Update the output buffer
    //
    atomicAdd(&output_buffer[screen_index].x,  result.x);
    atomicAdd(&output_buffer[screen_index].y,  result.y);
    atomicAdd(&output_buffer[screen_index].z,  result.z);
    atomicAdd(&output_buffer[screen_index].w,  samples_per_pass);

    atomicAdd(&output_buffer2[screen_index].x,  result.x * result.x);
    atomicAdd(&output_buffer2[screen_index].y,  result.y * result.y);
    atomicAdd(&output_buffer2[screen_index].z,  result.z * result.z);
    atomicAdd(&output_buffer2[screen_index].w,  samples_per_pass);

    //output_buffer[screen_index] = make_float4( result / samples_per_pass, 1.0 );
    //output_buffer2[screen_index] = make_float4( result * result / samples_per_pass, 1.0 );
}

rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );


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


////-----------------------------------------------------------------------------
////
////  Miss program
////
////-----------------------------------------------------------------------------
//
//rtDeclareVariable(float3, bg_color, , );
//
//RT_PROGRAM void miss()
//{
//    current_prd.radiance = bg_color;
//    current_prd.done = true;
//    current_prd.t = 1000;
//    current_prd.isMissed = true;
//}
//
//
