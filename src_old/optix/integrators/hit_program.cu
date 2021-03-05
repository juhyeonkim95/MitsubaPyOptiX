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

#include <optixu/optixu_math_namespace.h>
//#include "optix/common/prd_struct.h"
//#include "random.h"
//#include "helpers.h"
//#include "prd_struct.h"
#include "optix/q_table/qTable.cuh"
#include "optix/light/direct_light.h"
#include "optix/bsdf/bsdf.h"
#include "optix/bsdf/disney.h"

using namespace optix;

rtDeclareVariable( float3, shading_normal, attribute shading_normal, );
rtDeclareVariable( float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable( float3, front_hit_point, attribute front_hit_point, );
rtDeclareVariable( float3, back_hit_point, attribute back_hit_point, );
rtDeclareVariable( float3, texcoord, attribute texcoord, );


rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_pathtrace, prd, rtPayload, );

// rtDeclareVariable(rtObject, top_object, , );
// rtDeclareVariable(float, scene_epsilon, , );

//rtBuffer< rtCallableProgramId<float(float3 &ffnormal, float3 &wi)> > sysBRDFPdf;
//rtBuffer< rtCallableProgramId<float3(float3 &ffnormal, float3 &wo, PerRayData_pathtrace &prd)> > sysBRDFSample;
//rtBuffer< rtCallableProgramId<float3(float3 &mat_color, float3 &ffnormal, float3 &wo, float3 &new_direction)> > sysBRDFEval;
//rtBuffer< rtCallableProgramId<void(LightParameter &light, PerRayData_radiance &prd, LightSample &sample)> > sysLightSample;

//rtBuffer<MaterialParameter> sysMaterialParameters;
//rtDeclareVariable(int, materialId, , );
// rtDeclareVariable(int, programId, , );
//rtDeclareVariable(int, sysNumberOfLights, , );

// rtBuffer<ParallelogramLight> lights;


//rtDeclareVariable(float3,     diffuse_color, , );
//rtDeclareVariable(int,     diffuse_map_id, , );
rtDeclareVariable(unsigned int,     use_mis, , );
rtDeclareVariable(unsigned int,     sample_type, , );
rtDeclareVariable(unsigned int,     is_first_pass, , );

rtDeclareVariable(int, materialId, , );
rtBuffer<MaterialParameter> sysMaterialParameters;
rtDeclareVariable(float3, color0, , );
rtDeclareVariable(float3, color1, , );
rtDeclareVariable(Matrix3x3,  to_uv, , );
rtDeclareVariable(int, hasCheckerboard, , );


RT_PROGRAM void closest_hit()
{
	const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	const float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
	float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

	MaterialParameter mat = sysMaterialParameters[materialId];
    float3 normal = world_shading_normal;
	if (mat.isTwosided){
	    normal = ffnormal;
	}

    if (hasCheckerboard){
        float3 texcoordTransformed = to_uv * texcoord;
        float u = texcoordTransformed.x - int(texcoordTransformed.x);
        float v = texcoordTransformed.y - int(texcoordTransformed.y);
        if ((u > 0.5 && v > 0.5) || (u < 0.5 && v < 0.5)){
            mat.albedo = color0;
        } else {
            mat.albedo = color1;
        }
        //mat.albedo = make_float3(1,0,0);
    }
	if (mat.albedoID != RT_TEXTURE_ID_NULL)
	{
		const float3 texColor = make_float3(optix::rtTex2D<float4>(mat.albedoID , texcoord.x, 1 - texcoord.y));
		mat.albedo = make_float3(powf(texColor.x, 2.2f), powf(texColor.y, 2.2f), powf(texColor.z, 2.2f));
	}

    // Calculate anisotropic roughness along the tangent and bitangent directions
    float aspect = sqrt(1.0 - mat.anisotropic * 0.9);
    mat.ax = max(0.001, mat.roughness / aspect);
    mat.ay = max(0.001, mat.roughness * aspect);

	// prd.radiance += mat.emission * prd.attenuation;

	// TODO: Clean up handling of specular bounces
	// prd.specularBounce = mat.brdf == GLASS? true : false;
    float3 wi = -ray.direction;

    float3 hit_point = ray.origin + t_hit * ray.direction;

    float ior = mat.intIOR / mat.extIOR;

    optix::Onb onb( normal );
    float3 wi_local = to_local(onb, wi);

    State state;
    state.mat = mat;
    state.bitangent = onb.m_binormal;
    state.tangent = onb.m_tangent;
    state.normal = normal;
    state.ffnormal = ffnormal;

    // exiting : eta = ior
    // entering : eta = 1/ior
    state.eta = dot(state.normal, -ray.direction) > 0.0 ? (1.0 / ior) : ior;

    // Sample_info sample_info = sampleScatteringDirectionProportionalToQ();
    //MaterialParameter mat; // = sysMaterialParameters[materialId];
    //mat.diffuse_color = mat_color;
    // prd.mat = mat;

    float3 new_direction;
    float pdf;
    float3 weight;
    BSDFSample3f bs;

    // Disney
    if(programId == 99){
        Sample_info sample_info;
        weight = disney::DisneySample(state, wi, ffnormal, prd.seed, sample_info.direction, sample_info.pdf);
        new_direction = sample_info.direction;
        pdf = sample_info.pdf;
        weight = weight * abs(dot(ffnormal, sample_info.direction)) / pdf;
        prd.brdf_scatter_count += 1;
    }
    // BRDF Sampling
    else if (is_first_pass == 1 || (sample_type == 0 || sample_type == 1)){
        bs = Sample(mat, normal, wi_local, prd.seed);
        onb.inverse_transform(bs.wo);
        new_direction = bs.wo;
        weight = bs.weight;
        pdf = bs.pdf;
        prd.brdf_scatter_count += 1;
    }
    else if(programId == 0){
        Sample_info sample_info;

        if (sample_type == 2 || sample_type == 3){
            sample_info = sampleScatteringDirectionProportionalToQ(hit_point, normal, sample_type == 3, prd.seed);
            prd.q_scatter_count += 1;
        }
        else if(sample_type == 4){
            sample_info = sampleScatteringDirectionProportionalToQMCMC(hit_point, normal, prd.seed);
        }
        else if(sample_type == 5){
            sample_info = sampleScatteringDirectionProportionalToQReject(hit_point, normal, prd.seed);
        }

        new_direction = sample_info.direction;
        pdf = sample_info.pdf;

        float3 wo_local = to_local(onb, new_direction);
        float3 f = Eval(mat, normal, wi_local, wo_local);
        weight = make_float3(abs(f.x), abs(f.y), abs(f.z)) / pdf;

    } else {
        bs  = Sample(mat, normal, wi_local, prd.seed);
        onb.inverse_transform(bs.wo);
        new_direction = bs.wo;
        weight = bs.weight;
        pdf = bs.pdf;
        prd.brdf_scatter_count += 1;
    }
    if((use_mis == 1) && (bs.sampledLobe & BSDFLobe::SmoothLobe)){
        //Direct light Sampling
        prd.radiance = DirectLight(mat, ffnormal, hit_point, wi, prd.seed);
        // prd.result += prd.attenuation * DirectLight(mat, ffnormal, hit_point, prd);
    }

//    if(sample_type == 0 || sample_type == 1){
//
//        if(programId == 99){
//            Sample_info sample_info;
//            weight = disney::DisneySample(state, wi, ffnormal, prd.seed, sample_info.direction, sample_info.pdf);
//            new_direction = sample_info.direction;
//            pdf = sample_info.pdf;
//            weight = weight * abs(dot(ffnormal, sample_info.direction)) / pdf;
//        } else {
//            bs  = Sample(mat, normal, wi_local, prd.seed);
//            // BRDF Sampling
//            new_direction = bs.wo.x * onb.m_tangent +  bs.wo.y * onb.m_binormal + bs.wo.z * onb.m_normal;
//            weight = bs.weight;
//            pdf = bs.pdf;
//        }
//
//        if(use_mis == 1 && (bs.sampledLobe & BSDFLobe::SmoothLobe)){
//            //Direct light Sampling
//            prd.radiance = DirectLight(mat, normal, hit_point, wi, prd.seed);
//            // prd.result += prd.attenuation * DirectLight(mat, ffnormal, hit_point, prd);
//        }
//    }
//    else if (sample_type == 2 || sample_type == 3){
//        if(programId == 0){
//
//        }
//        else if(programId == 99){
//            Sample_info sample_info;
//            weight = disney::DisneySample(state, wi, ffnormal, prd.seed, sample_info.direction, sample_info.pdf);
//            new_direction = sample_info.direction;
//            pdf = sample_info.pdf;
//            weight = weight * abs(dot(ffnormal, sample_info.direction)) / pdf;
//        } else {
//            bs  = Sample(mat, normal, wi_local, prd.seed);
//            // BRDF Sampling
//            new_direction = bs.wo.x * onb.m_tangent +  bs.wo.y * onb.m_binormal + bs.wo.z * onb.m_normal;
//            weight = bs.weight;
//            pdf = bs.pdf;
//        }
//    }
//    else if (sample_type == 4){
//        if(rnd(prd.seed) > 0.0){
//            ///Sample_info sample_info = sampleScatteringDirectionProportionalToQSphere(hit_point, prd.seed);
//            //Sample_info sample_info = sampleScatteringDirectionProportionalToQ(hit_point, ffnormal, false, prd.seed);
//            Sample_info sample_info = sampleScatteringDirectionProportionalToQMCMC(hit_point, normal, prd.seed);
//
//            new_direction = sample_info.direction;
//            pdf = sample_info.pdf;
//            if(dot(new_direction, normal) <= 0.0){
//                pdf = 0.0;
//            }
//        } else {
//            // BRDF Sampling
//            new_direction = Sample(mat, normal, wo, prd.seed);
//            pdf = Pdf(mat, normal, wo, new_direction);
//        }
//    }
//    else if (sample_type == 5){
//        Sample_info sample_info = sampleScatteringDirectionProportionalToQReject(hit_point, normal, prd.seed);
//        new_direction = sample_info.direction;
//        pdf = sample_info.pdf;
//        if(dot(new_direction, normal) <= 0.0){
//            pdf = 0.0;
//        }
//    }
    // pdf = max(pdf, 0.01);

    // float3 f = Eval(mat, normal, wo, new_direction);

    prd.t = t_hit;
    prd.direction = new_direction;
    prd.origin = hit_point;

    prd.scatterPdf = pdf;
    prd.isSpecular = (bs.sampledLobe & BSDFLobe::SpecularLobe);
	if (pdf <= 0.0f || dot(weight, weight) == 0.0){
	    prd.current_attenuation = make_float3(0.0f);
	    prd.done = true;
	}
	else{
	    prd.current_attenuation = weight;
	}
	prd.diffuse_color = mat.albedo;
    prd.normal = normal;
    prd.material_type = programId;
}