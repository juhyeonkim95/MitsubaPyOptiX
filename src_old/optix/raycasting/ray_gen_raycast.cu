/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include "optix/raycasting/raycast_construct.h"
#include "optix/common/random.h"
#include "optix/cameras/camera.h"

rtDeclareVariable( Hit, hit_prd, rtPayload, );

//------------------------------------------------------------------------------
//
// Ray generation
//
//------------------------------------------------------------------------------

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(unsigned int, depth, , );
rtDeclareVariable(unsigned int, frame_number, , );

rtDeclareVariable(rtObject, top_object, , );

rtBuffer<Hit, 2>  hits;
rtBuffer<RayData, 2>  rays;

RT_PROGRAM void ray_gen()
{
    size_t2 screen = rays.size();
    Hit hit_prd = hits[launch_index];
    RayData ray = rays[launch_index];

    if(depth == 0){
        unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame_number);

        float2 jitter = make_float2(rnd(seed)-0.5, rnd(seed) - 0.5);
        float2 inv_screen = 1.0f/make_float2(screen) * 2.f;
        float2 d = (make_float2(launch_index) + jitter) * inv_screen - 1.f;
        generate_ray(d, ray.origin, ray.dir, seed);

        hit_prd.t           = -1.0f;
        hit_prd.geom_normal = optix::make_float3(0, 0, 1);
        hit_prd.hit_point = optix::make_float3(0, 0, 0);
        hit_prd.color = optix::make_float3(0, 0, 0);
        hit_prd.attenuation = optix::make_float3(1, 1, 1);
        hit_prd.new_direction = optix::make_float3(0, 0, 1);
        hit_prd.pdf = 1.0f;
        hit_prd.done = 0;
        hit_prd.seed = seed;
        hit_prd.result = optix::make_float3(0, 0, 0);
    }

    if(hit_prd.done == 1){
        return;
    }


    rtTrace( top_object,
             optix::make_Ray( ray.origin, ray.dir, 0, 1e-3, RT_DEFAULT_MAX ),
             hit_prd );

    hits[ launch_index ] = hit_prd;
}

//------------------------------------------------------------------------------
//
// Exception program for debugging only
//
//------------------------------------------------------------------------------


RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d)\n", code, launch_index );
  Hit hit_prd;
  hit_prd.t           = -1.0f;
  hit_prd.geom_normal = optix::make_float3(1, 0, 0);
  hit_prd.hit_point = optix::make_float3(1, 0, 0);
  hit_prd.color = optix::make_float3(0, 0, 0);
  hit_prd.result = optix::make_float3(1, 0, 0);
  hits[ launch_index ] = hit_prd;
}