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
#include "optixPathTracer.h"
#include "random.h"
#include "helpers.h"
#include "prd_struct.h"
#include "qTable.cuh"


using namespace optix;

//rtBuffer<float, 2>              q_table;
//rtBuffer<float, 2>              q_table_old;
//rtBuffer<float, 2>              v_table;
//rtBuffer<float, 2>              v_table_old;
//rtBuffer<uint, 2>               visit_counts;
//rtBuffer<float3, 1>             unitUVVectors;

// Scene wide variables
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(uint2,         launch_index, rtLaunchIndex, );

rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );

//const float sigma_s = 0.0009;
//const float sigma_a = 0.0006;
//const float sigma_t = 0.0015;
const float sigma_s = 0.0;
const float sigma_a = 0.0000;
const float sigma_t = 0.0;
rtDeclareVariable(unsigned int,     scatter_sample_type, , );

//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(unsigned int,  frame_number, , );
rtDeclareVariable(unsigned int,  sqrt_num_samples, , );
rtDeclareVariable(unsigned int,  rr_begin_depth, , );
rtDeclareVariable(unsigned int,  pathtrace_ray_type, , );
rtDeclareVariable(unsigned int,  pathtrace_shadow_ray_type, , );

rtBuffer<float4, 2>              output_buffer;
rtBuffer<float4, 2>              output_buffer2;
rtBuffer<float, 2>               hit_count_buffer;
rtBuffer<float, 2>               path_length_buffer;


rtBuffer<ParallelogramLight>     lights;
//rtBuffer<LightParameter> sysLightParameters;

RT_PROGRAM void pathtrace_camera()
{
    size_t2 screen = output_buffer.size();
    float2 inv_screen = 1.0f/make_float2(screen) * 2.f;
    float2 pixel = (make_float2(launch_index)) * inv_screen - 1.f;

    float2 jitter_scale = inv_screen / sqrt_num_samples;
    unsigned int samples_per_pixel = 1;//sqrt_num_samples*sqrt_num_samples;
    float3 result = make_float3(0.0f);
    float hit_count = 0;
    unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame_number);
    do
    {
        //
        // Sample pixel using jittering
        //
        unsigned int x = samples_per_pixel%sqrt_num_samples;
        unsigned int y = samples_per_pixel/sqrt_num_samples;
        float2 jitter = make_float2(x-rnd(seed), y-rnd(seed));
        float2 d = pixel + jitter*jitter_scale;
        float3 ray_origin = eye;
        float3 ray_direction = normalize(d.x*U + d.y*V + W);

        // Initialze per-ray data
        PerRayData_pathtrace prd;
        prd.result = make_float3(0.f);
        prd.radiance = make_float3(0.f);
        prd.attenuation = make_float3(1.f);
        prd.countEmitted = true;
        prd.done = false;
        prd.seed = seed;
        prd.depth = 0;
        prd.normal = make_float3(0, 1, 0);

        // Each iteration is a segment of the ray path.  The closest hit will
        // return new segments to be traced here.
        for(;;)
        {
            prd.current_attenuation = make_float3(1.f);
            Ray ray = make_Ray(ray_origin, ray_direction, pathtrace_ray_type, scene_epsilon, RT_DEFAULT_MAX);
            rtTrace(top_object, ray, prd);

            if(sigma_t > 0.0){
                float tout = prd.t;
                float tin = scene_epsilon;
                float prob_s =  1 - exp(-sigma_s * (tout - tin));
                float scaleBy = 1.0/(1.0-prob_s);

                // scatter occurs.
                if(rnd(prd.seed) < prob_s){
                    float s = sampleSegment(rnd(seed), sigma_s, tout - tin);
                    float3 x = ray.origin + ray.direction * (tin + s);

                    float3 dir;
                    if(scatter_sample_type == 0){
                        dir = sampleSphere(rnd(prd.seed), rnd(prd.seed));
                        prd.attenuation *= HG_phase_function(-0.5, dot(dir, make_float3(0,0,1)));
                        float3 u, v;
                        generateOrthoBasis(u, v, ray.direction);
                        dir = u*dir.x+v*dir.y+ray.direction*dir.z;
                    }else if(scatter_sample_type ==1){
                        dir = sampleHG(-0.5, rnd(prd.seed), rnd(prd.seed));
                        float3 u, v;
                        generateOrthoBasis(u, v, ray.direction);
                        dir = u*dir.x+v*dir.y+ray.direction*dir.z;
                    }else if(scatter_sample_type == 2){
                        Sample_info sample_info = sampleScatteringDirectionProportionalToQVolume(x, prd.seed);
                        dir = sample_info.direction;
                        float p_w = sample_info.p_w;
                        float a = HG_phase_function(-0.5, dot(dir, ray.direction)) / (p_w * 2 * float(unitUVNumber.x * unitUVNumber.y));
                        prd.attenuation *= a;
                    }else if(scatter_sample_type == 3){
                        Sample_info sample_info = sampleScatteringDirectionProportionalToQVolumeHG(x, ray.direction, -0.5, prd.seed);
                        dir = sample_info.direction;
                        float p_w = sample_info.p_w;
                        float a = HG_phase_function(-0.5, dot(dir, ray.direction)) / (p_w * 2 * float(unitUVNumber.x * unitUVNumber.y));
                        prd.attenuation *= a;
                    }


                    //optix::Onb onb( ray.direction );
                    //onb.inverse_transform( dir );
                    ray_origin = x;
                    ray_direction = dir;

                    prd.depth++;
                    prd.attenuation *= prob_s * (1.0/prob_s);
                    continue;

                }
                // scatter doesn't occur.
                else {
                    prd.attenuation *= exp(-sigma_t * tout);
                    //prd.attenuation *= prd.current_attenuation;
                }

                prd.attenuation *= scaleBy;
            }
            // prd.current_attenuation = make_float3(0.9f);
            prd.attenuation *= prd.current_attenuation;
            //prd.attenuation *= make_float3(0.5f);

            //float ms = scatterVolume();

            float reward = prd.radiance.x;
            float new_value;
            if(prd.done){
                new_value = reward * 0.01f;
            }else{
                //new_value = reward + 0.9f * getQValue(prd.origin, prd.direction);
                float f_s = (prd.diffuse_color.x + prd.diffuse_color.y + prd.diffuse_color.z) / 3.0f;
                float nextQValue = getNextQValue(prd.origin, prd.normal, prd.direction) * f_s;
            }

            if(prd.depth > 0){
                float alpha = 1.0f / sqrt(1.0f + updateVisit(ray_origin, ray_direction));
                //float alpha = 0.01f;
                float update_value = (1-alpha) * getQValue(ray_origin, ray_direction) + alpha * new_value;
                setQValue(ray_origin, ray_direction, update_value);
            }

            if(prd.done)
            {
                // We have hit the background or a luminaire
                prd.result += prd.radiance * prd.attenuation;
                // prd.result += prd.current_attenuation;

                if(prd.result.x > 0)
                    hit_count += 1;
                //prd.result = make_float3(getQValue(ray_origin, ray_direction));
                break;
            }
            // Russian roulette termination
            if(prd.depth >= rr_begin_depth)
            {
                float pcont = fmaxf(prd.attenuation);
                if(rnd(prd.seed) >= pcont)
                    break;
                prd.attenuation /= pcont;
            }

            if(prd.depth >= 100){
                break;
            }

            prd.depth++;
            prd.result += prd.radiance * prd.attenuation;
            //rtPrintf("attenuation %f %f %f\n", prd.attenuation.x, prd.attenuation.y, prd.attenuation.z);

            // Update ray data for the next path segment
            ray_origin = prd.origin;
            ray_direction = prd.direction;
        }

        result += prd.result;
        seed = prd.seed;

        hit_count_buffer[launch_index] += hit_count;
        path_length_buffer[launch_index] += float(prd.depth);

    } while (--samples_per_pixel);

    //
    // Update the output buffer
    //
    float3 pixel_color = result;///(sqrt_num_samples*sqrt_num_samples);
    output_buffer[launch_index] += make_float4( pixel_color, 1.0f );
    output_buffer2[launch_index] += make_float4( pixel_color * pixel_color, 1.0f );

}


//-----------------------------------------------------------------------------
//
//  Emissive surface closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,     diffuse_color, , );
rtDeclareVariable(int,     diffuse_map_id, , );

rtDeclareVariable(float3,     geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3,     shading_normal,   attribute shading_normal, );
rtDeclareVariable(float3, texcoord, attribute texcoord, );

rtDeclareVariable(optix::Ray, ray,              rtCurrentRay, );
rtDeclareVariable(float,      t_hit,            rtIntersectionDistance, );
rtDeclareVariable(unsigned int,     sample_type, , );
rtDeclareVariable(float3,        emission_color, , );

//-----------------------------------------------------------------------------
//
//  Lambertian surface closest-hit
//
//-----------------------------------------------------------------------------


RT_PROGRAM void diffuseEmitter()
{
    current_prd.radiance = current_prd.countEmitted ? emission_color : make_float3(0.f);
    current_prd.done = true;
    current_prd.t = t_hit;
}

RT_PROGRAM void diffuse()
{
    float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
    float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

    float3 hitpoint = ray.origin + t_hit * ray.direction;

    //
    // Generate a reflection ray.  This will be traced back in ray-gen.
    //
    current_prd.origin = hitpoint;

    float3 diffuse_color_final;
    if (diffuse_map_id != RT_TEXTURE_ID_NULL){
        diffuse_color_final = make_float3(optix::rtTex2D<float4>(diffuse_map_id, texcoord.x, 1 - texcoord.y));
    }
    else{
        diffuse_color_final = diffuse_color;
    }
    // diffuse_color_final = diffuse_color;
    // NOTE: f/pdf = 1 since we are perfectly importance sampling lambertian
    // with cosine density.
    //rtPrintf("color : %f %f %f\n", diffuse_color.x, diffuse_color.y, diffuse_color.z);
    if(sample_type == 0){
        // CASE 1. uniform sampling
        float z1=rnd(current_prd.seed);
        float z2=rnd(current_prd.seed);
        float3 p = UniformSampleSphere(z1, z2);
        optix::Onb onb( ffnormal );
        onb.inverse_transform( p );
        current_prd.direction = p;
        // A = f_s * cos(n, w) / p_w
        // f_s = diffuse_color / pi
        // p_w = 1 / (2 * pi)
        // --> A = 2 * diffuse_color * cos(n, w)
        current_prd.current_attenuation = 2 * diffuse_color_final * dot(ffnormal, p);
    }
    else if(sample_type == 1){
        // CASE 2. cosine sampling
        float z1=rnd(current_prd.seed);
        float z2=rnd(current_prd.seed);
        float3 p;
        cosine_sample_hemisphere(z1, z2, p);
        optix::Onb onb( ffnormal );
        onb.inverse_transform( p );
        current_prd.direction = p;

        // A = f_s * cos(n, w) / p_w
        // f_s = diffuse_color / pi
        // p_w = cos(n, w) / pi
        // --> A = diffuse_color
        current_prd.current_attenuation = diffuse_color_final;
    }
    else if(sample_type == 2 || sample_type == 3){
        bool considerCosineTerm = (sample_type == 3);
        //CASE 3. Q proportion sampling
        Sample_info sample_info = sampleScatteringDirectionProportionalToQ(hitpoint, ffnormal, considerCosineTerm, current_prd.seed);
        current_prd.direction = sample_info.direction;
        float p_w = sample_info.p_w;
        // A = f_s * cos(n, w) / p_w
        // f_s = diffuse_color / pi
        // p_w = p_w_0 / (2 * pi) * 16
        // --> A = diffuse_color
        float temp1 = dot(ffnormal, current_prd.direction);
        if(temp1<0)
            temp1=0;
        float tt = (p_w * float(unitUVNumber.x * unitUVNumber.y));
        float a = temp1 / tt;
        if(a > 10){a=10;}

        a = dot(ffnormal, current_prd.direction) / (p_w * float(unitUVNumber.x * unitUVNumber.y));
        current_prd.current_attenuation = diffuse_color_final * 2 * a;
    }


    //
    // Next event estimation (compute direct lighting).
    //
//    unsigned int num_lights = lights.size();
//    float3 result = make_float3(0.0f);
//
//    for(int i = 0; i < num_lights; ++i)
//    {
//        // Choose random point on light
//        ParallelogramLight light = lights[i];
//        const float z1 = rnd(current_prd.seed);
//        const float z2 = rnd(current_prd.seed);
//        const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;
//
//        // Calculate properties of light sample (for area based pdf)
//        const float  Ldist = length(light_pos - hitpoint);
//        const float3 L     = normalize(light_pos - hitpoint);
//        const float  nDl   = dot( ffnormal, L );
//        const float  LnDl  = dot( light.normal, L );
//
//        // cast shadow ray
//        if ( nDl > 0.0f && LnDl > 0.0f )
//        {
//            PerRayData_pathtrace_shadow shadow_prd;
//            shadow_prd.inShadow = false;
//            // Note: bias both ends of the shadow ray, in case the light is also present as geometry in the scene.
//            Ray shadow_ray = make_Ray( hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, Ldist - scene_epsilon );
//            rtTrace(top_object, shadow_ray, shadow_prd);
//
//            if(!shadow_prd.inShadow)
//            {
//                const float A = length(cross(light.v1, light.v2));
//                // convert area based pdf to solid angle
//                const float weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
//                result += light.emission * weight;
//            }
//        }
//    }
//
//    current_prd.radiance = result;

    //current_prd.current_attenuation = make_float3(0.1f);
    current_prd.t = t_hit;
    current_prd.normal = ffnormal;
    current_prd.diffuse_color = diffuse_color_final;
}

RT_PROGRAM void metal()
{
    float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
    float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
    float3 hitpoint = ray.origin + t_hit * ray.direction;
    //
    // Generate a reflection ray.  This will be traced back in ray-gen.
    //
    current_prd.origin = hitpoint;
    current_prd.direction = optix::reflect(ray.direction, ffnormal);
    current_prd.current_attenuation = make_float3(0.8, 0.6, 0.2);
    current_prd.t = t_hit;
}



rtDeclareVariable(float,        refraction_index, , );
rtDeclareVariable(float3,       refraction_color, , );
rtDeclareVariable(float3,       reflection_color, , );
rtDeclareVariable(float3,       extinction, , );

RT_PROGRAM void glass()
{
    const float3 w_out = -ray.direction;
    float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float cos_theta_i = optix::dot( w_out, normal );
    float3 hitpoint = ray.origin + t_hit * ray.direction;

    float eta;
    float3 transmittance = make_float3( 1.0f );
    if( cos_theta_i > 0.0f ) {
        // Ray is entering
        eta = refraction_index;  // Note: does not handle nested dielectrics
    } else {
        // Ray is exiting; apply Beer's Law.
        // This is derived in Shirley's Fundamentals of Graphics book.
        transmittance = optix::expf( -extinction * t_hit );
        eta         = 1.0f / refraction_index;
        cos_theta_i = -cos_theta_i;
        normal      = -normal;
    }

    float3 w_t;
    const bool tir           = !optix::refract( w_t, -w_out, normal, eta );

    const float cos_theta_t  = -optix::dot( normal, w_t );
    const float R            = tir  ?
                               1.0f :
                               fresnel( cos_theta_i, cos_theta_t, eta );

    // Importance sample the Fresnel term
    const float z = rnd( current_prd.seed );
    if( z <= R ) {
        // Reflect
        const float3 w_in = optix::reflect( -w_out, normal );
        current_prd.origin = hitpoint;
        current_prd.direction = w_in;
        current_prd.current_attenuation = reflection_color*transmittance;
    } else {
        // Refract
        const float3 w_in = w_t;
        current_prd.origin = hitpoint;
        current_prd.direction = w_in;
        current_prd.current_attenuation = refraction_color*transmittance;
    }
    current_prd.t = t_hit;

}
//-----------------------------------------------------------------------------
//
//  Shadow any-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(PerRayData_pathtrace_shadow, current_prd_shadow, rtPayload, );

RT_PROGRAM void shadow()
{
    current_prd_shadow.inShadow = true;
    rtTerminateRay();
}


//-----------------------------------------------------------------------------
//
//  Exception program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void exception()
{
    output_buffer[launch_index] = make_float4(bad_color, 1.0f);
}


//-----------------------------------------------------------------------------
//
//  Miss program
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, bg_color, , );

RT_PROGRAM void miss()
{
    current_prd.radiance = bg_color;
    current_prd.done = true;
    current_prd.t = 1000;
}


