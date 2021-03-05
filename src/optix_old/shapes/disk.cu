/*
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

#include <optix_world.h>

using namespace optix;

rtDeclareVariable(float4,  disk_pos_radii, , );
rtDeclareVariable(float3,  disk_normal, , );


rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(int, hitTriIdx,  attribute hitTriIdx, );


RT_PROGRAM void intersect(int primIdx)
{
    float3 position = make_float3(disk_pos_radii);
    float radius = disk_pos_radii.w;
    float3 normal = disk_normal;

    float denom = dot(normal, ray.direction);
    if (denom != 0.0f)
    {
        const float3 p = position - ray.origin;
        const float t = dot(p, normal) / denom;

        const float3 hit = ray.origin + t * ray.direction;
        const float3 d = hit - position;

        if (dot(d, d) < radius * radius)
        {
            if (rtPotentialIntersection(t))
            {
                hitTriIdx = 0;
                shading_normal = geometric_normal = normal;
                rtReportIntersection(0);
            }
        }
    }
}

RT_PROGRAM void bounds (int, float result[6])
{
    float3 position = make_float3(disk_pos_radii);
    float3 radius = make_float3(disk_pos_radii.w);

    optix::Aabb* aabb = (optix::Aabb*)result;

    if( radius.x > 0.0f  && !isinf(radius.x) ) {
        aabb->m_min = position - radius;
        aabb->m_max = position + radius;
    } else {
        aabb->invalidate();
    }
}

