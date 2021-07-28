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
#include "optix/common/prd_struct.h"
#include "optix/common/helpers.h"
//#include "optix/q_table/radiance_record.cuh"
//#include "optix/q_table/qTable.cuh"
//#include "optix/light/direct_light.h"
//#include "optix/bsdf/bsdf.h"
//#include "optix/bsdf/disney.h"
//#include "optix/app_config.h"
//#include "optix/texture/texture.h"
//#include "optix/utils/material_value_loader.h"

using namespace optix;

rtDeclareVariable( float3, shading_normal, attribute shading_normal, );
rtDeclareVariable( float3, geometric_normal, attribute geometric_normal, );
//rtDeclareVariable( float3, geometric_normal, attribute geometric_normal, );
//rtDeclareVariable( float3, bitangent, attribute bitangent, );
//rtDeclareVariable( float3, tangent, attribute tangent, );

rtDeclareVariable( float3, front_hit_point, attribute front_hit_point, );
rtDeclareVariable( float3, back_hit_point, attribute back_hit_point, );
rtDeclareVariable( float3, texcoord, attribute texcoord, );


rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(SurfaceInteraction, si, rtPayload, );

//rtBuffer< rtCallableProgramId<float(float3 &ffnormal, float3 &wi)> > sysBRDFPdf;
//rtBuffer< rtCallableProgramId<float3(float3 &ffnormal, float3 &wo, PerRayData_pathtrace &prd)> > sysBRDFSample;
//rtBuffer< rtCallableProgramId<float3(float3 &mat_color, float3 &ffnormal, float3 &wo, float3 &new_direction)> > sysBRDFEval;
//rtBuffer< rtCallableProgramId<void(LightParameter &light, PerRayData_radiance &prd, LightSample &sample)> > sysLightSample;


//rtDeclareVariable(unsigned int,     use_mis, , );
//rtDeclareVariable(unsigned int,     sample_type, , );
//rtDeclareVariable(unsigned int,     is_first_pass, , );
//rtDeclareVariable(float,     bsdf_sampling_fraction, , );


rtBuffer<MaterialParameter> sysMaterialParameters;
rtDeclareVariable(int, materialId, , );

RT_PROGRAM void closest_hit()
{
    // Transform normal from object to world coordinate
	const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	const float3 world_geometric_normal = normalize(rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );

    // Face forwarding normal (ffnormal dot ray_direction > 0)
    float3 ff_normal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

    // Material parameter
	MaterialParameter& mat = sysMaterialParameters[materialId];
    float3 normal = mat.isTwosided? ff_normal : world_shading_normal;

    si.material_id = materialId;
    si.normal = normal;
    si.uv = texcoord;

    float3 wi = -ray.direction;
    optix::Onb onb( normal );

    si.wi = to_local(onb, wi);
    si.p = ray.origin + t_hit * ray.direction;
}

//RT_PROGRAM void closest_hit()
//{

//
//	if (mat.bumpID != RT_TEXTURE_ID_NULL){
//        // TODO: Implement normal map
//
//	}
//
//    float3 wi = -ray.direction;
//    prd.origin = ray.origin + t_hit * ray.direction;
//
//    optix::Onb onb( normal );
//    float3 wi_local = to_local(onb, wi);
//
//    float3 new_direction;
//    float pdf = 0;
//    float3 weight = make_float3(0.0f);
//    BSDFSample3f bs;
//
//#if OPTIMIZE_HIT_PROGRAM
//    SurfaceInteraction si;
//    si.wi = wi_local;
//    si.uv = texcoord;
//    bs = bsdf::Sample(mat, si, prd.seed);
//    onb.inverse_transform(bs.wo);
//    new_direction = bs.wo;
//    weight = bs.weight;
//    pdf = bs.pdf;
//#else
//    bool q_sample_implemented_type = (programId == 0);
//    bool need_brdf_sampling = is_first_pass || (sample_type == SAMPLE_BSDF) || !q_sample_implemented_type;
//    bool need_uniform_sampling = (sample_type == 0 && programId == 0);
//
//    // need_brdf_sampling |= (rnd(prd.seed) < bsdf_sampling_fraction);
//    // need_brdf_sampling |= (sample_type == 7);
//    bool is_material_specular = (programId == 1) | (programId == 3);
//
//    // 0. uniform sampling
//    if(need_uniform_sampling){
//        new_direction = UniformSampleHemisphere(rnd(prd.seed), rnd(prd.seed));
//        pdf = 1 / (2 * M_PIf);
//        //new_direction = UniformSampleSphere(rnd(prd.seed), rnd(prd.seed));
//        //pdf = 1 / (4 * M_PIf);
//
//        float3 f = bsdf::Eval(albedo, mat, wi_local, new_direction);
//        weight = make_float3(abs(f.x), abs(f.y), abs(f.z)) / pdf;
//        onb.inverse_transform(new_direction);
//    }
void sample()
{

}
//    // 1. BSDF sampling (prop to f * cos)
//    else if(need_brdf_sampling){
//        bs = bsdf::Sample(albedo, mat, wi_local, prd.seed);
//        onb.inverse_transform(bs.wo);
//        new_direction = bs.wo;
//        weight = bs.weight;
//        pdf = bs.pdf;
//        // prd.brdf_scatter_count += 1;
//    }
//    // 2. BSDF & Q sampling (prop to f * cos and Q)
//    else if((sample_type == SAMPLE_QUADTREE)|| (sample_type == SAMPLE_SPHERICAL_INV)){
//        // Use Multiple Importance Sampling
//        float bsdf_pdf;
//        float radiance_pdf;
//
//        // (1) BSDF Sampling (prop to f * cos)
//        if (rnd(prd.seed) < bsdf_sampling_fraction){
//            bs = bsdf::Sample(albedo, mat, wi_local, prd.seed);
//            onb.inverse_transform(bs.wo);
//            new_direction = bs.wo;
//            weight = bs.weight;
//            bsdf_pdf = bs.pdf;
//            // prd.brdf_scatter_count += 1;
//
//            weight *= bsdf_pdf;
//            radiance_pdf = radiance_record::Pdf(prd.origin, normal, new_direction, sample_type);
//        }
//
//        // (2) Radiance Sampling (prop to Q)
//        else {
//            Sample_info sample_info = radiance_record::Sample(prd.origin, normal, prd.seed, sample_type);
//            // Sample_info sample_info = sampleScatteringDirectionProportionalToQQuadTree(prd.origin, normal, prd.seed);
//            new_direction = sample_info.direction;
//            radiance_pdf = sample_info.pdf;
//            float3 wo_local = to_local(onb, new_direction);
//            weight = bsdf::Eval(albedo, mat, wi_local, wo_local);
//
//            bsdf_pdf = bsdf::Pdf(mat, wi_local, wo_local);
//        }
//
//        // invalid sample
//        if(dot(weight, weight) == 0.0){
//            increment_invalid_sample(prd.origin);
//            if(prd.depth == 0)
//                prd.invalid_scatter_count = 1;
//        } else {
//            increment_valid_sample(prd.origin);
//            if(prd.depth == 0)
//                prd.valid_scatter_count = 1;
//        }
//
//        // Apply MIS
//        pdf = bsdf_sampling_fraction * bsdf_pdf + (1 - bsdf_sampling_fraction) * radiance_pdf;
//        weight = weight / pdf;
//    }
//    else {
//        Sample_info sample_info;
//
//        if (sample_type == 2 || sample_type == 3){
//            sample_info = sampleScatteringDirectionProportionalToQ(prd.origin, normal, sample_type == 3, prd.seed);
//            // prd.q_scatter_count += 1;
//        }
//        else if(sample_type == 4){
//            sample_info = sampleScatteringDirectionProportionalToQMCMC(prd.origin, normal, prd.seed);
//        }
//        else if(sample_type == 5){
//            sample_info = sampleScatteringDirectionProportionalToQReject(prd.origin, normal, prd.seed);
//        }else if(sample_type == 8){
//            sample_info = sampleScatteringDirectionProportionalToQReject2(prd.origin, normal, prd.seed);
//        }
//        else if(sample_type == 6){
//            sample_info = sampleScatteringDirectionProportionalToQSphere(prd.origin, prd.seed);
//        }
//        else if(sample_type == 7){
//            sample_info = sampleScatteringDirectionProportionalToQQuadTree(prd.origin, normal, prd.seed);
//        }
//
//        new_direction = sample_info.direction;
//        pdf = sample_info.pdf;
//        if(pdf > 0 && dot(new_direction, normal) > 0.0f){
//            float3 wo_local = to_local(onb, new_direction);
//            float3 f = bsdf::Eval(albedo, mat, wi_local, wo_local);
//            weight = f / pdf;
//        }
//
//        // invalid sample
//        if(dot(weight, weight) == 0.0){
//            increment_invalid_sample(prd.origin);
//            if(prd.depth == 0)
//                prd.invalid_scatter_count = 1;
//        } else {
//            increment_valid_sample(prd.origin);
//            if(prd.depth == 0)
//                prd.valid_scatter_count = 1;
//        }
//    }
//#endif
//
//
//    prd.t = t_hit;
//    prd.direction = new_direction;
//    //prd.origin = hit_point;
//
//    prd.scatterPdf = pdf;
//    prd.isSpecular = (bs.sampledLobe & BSDFLobe::SpecularLobe);
//	if (pdf <= 0.0f || dot(weight, weight) == 0.0){
//	    prd.current_attenuation = make_float3(0.0f);
//	    prd.done = true;
//	}
//	else{
//	    prd.current_attenuation = weight;
//	}
//	prd.diffuse_color = make_float3(1.0);
//    prd.normal = normal;
//    prd.material_type = programId;
//}