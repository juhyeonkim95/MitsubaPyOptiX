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

using namespace optix;

rtDeclareVariable( float3, shading_normal, attribute shading_normal, );
rtDeclareVariable( float3, geometric_normal, attribute geometric_normal, );
//rtDeclareVariable( float3, front_hit_point, attribute front_hit_point, );
//rtDeclareVariable( float3, back_hit_point, attribute back_hit_point, );
rtDeclareVariable( float3, texcoord, attribute texcoord, );


rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_pathtrace, prd, rtPayload, );
rtDeclareVariable(PerRayData_pathtrace_shadow, prd_shadow, rtPayload, );

// rtDeclareVariable(rtObject, top_object, , );
// rtDeclareVariable(float, scene_epsilon, , );

rtBuffer< rtCallableProgramId<float(float3 &ffnormal, float3 &wi)> > sysBRDFPdf;
rtBuffer< rtCallableProgramId<float3(float3 &ffnormal, float3 &wo, PerRayData_pathtrace &prd)> > sysBRDFSample;
rtBuffer< rtCallableProgramId<float3(float3 &mat_color, float3 &ffnormal, float3 &wo, float3 &new_direction)> > sysBRDFEval;
//rtBuffer< rtCallableProgramId<void(LightParameter &light, PerRayData_radiance &prd, LightSample &sample)> > sysLightSample;

rtBuffer<MaterialParameter> sysMaterialParameters;
//rtDeclareVariable(int, materialId, , );
//rtDeclareVariable(int, programId, , );
//rtDeclareVariable(int, sysNumberOfLights, , );

// rtBuffer<ParallelogramLight> lights;


rtDeclareVariable(float3,     diffuse_color, , );
rtDeclareVariable(int,     diffuse_map_id, , );
rtDeclareVariable(unsigned int,     use_mis, , );
//rtDeclareVariable(unsigned int,     sample_type, , );

RT_PROGRAM void closest_hit()
{
	const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	const float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
	float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

	//MaterialParameter mat = sysMaterialParameters[materialId];
    float3 mat_color = diffuse_color;
	if (diffuse_map_id != RT_TEXTURE_ID_NULL)
	{
		const float3 texColor = make_float3(optix::rtTex2D<float4>(diffuse_map_id, texcoord.x, 1 - texcoord.y));
		mat_color = make_float3(powf(texColor.x, 2.2f), powf(texColor.y, 2.2f), powf(texColor.z, 2.2f));
	}

	// prd.radiance += mat.emission * prd.attenuation;

	// TODO: Clean up handling of specular bounces
	// prd.specularBounce = mat.brdf == GLASS? true : false;
    float3 wo = -ray.direction;
    float3 hit_point = ray.origin + t_hit * ray.direction;


    // Sample_info sample_info = sampleScatteringDirectionProportionalToQ();
    MaterialParameter mat; // = sysMaterialParameters[materialId];
    mat.diffuse_color = mat_color;
    prd.mat = mat;

    float3 new_direction;
    float pdf;

    if(sample_type == 0 || sample_type == 1){
        if(use_mis == 1){
            //Direct light Sampling
            prd.radiance = DirectLight(mat, ffnormal, hit_point, wo, prd.seed);
            // prd.result += prd.attenuation * DirectLight(mat, ffnormal, hit_point, prd);
        }

        // BRDF Sampling
        new_direction = Sample(mat, ffnormal, wo, prd.seed);
        pdf = Pdf(mat, ffnormal, wo, new_direction);
    }
    else if (sample_type == 2 || sample_type == 3){
        if(use_mis == 1){
            //Direct light Sampling
            prd.radiance = DirectLight(mat, ffnormal, hit_point, wo, prd.seed);
            // prd.result += prd.attenuation * DirectLight(mat, ffnormal, hit_point, prd);
        }
        Sample_info sample_info = sampleScatteringDirectionProportionalToQ(hit_point, ffnormal, sample_type == 3, prd.seed);
        new_direction = sample_info.direction;
        pdf = sample_info.pdf;
    }
    else if (sample_type == 4){
        if(rnd(prd.seed) > 0.0){
            ///Sample_info sample_info = sampleScatteringDirectionProportionalToQSphere(hit_point, prd.seed);
            //Sample_info sample_info = sampleScatteringDirectionProportionalToQ(hit_point, ffnormal, false, prd.seed);
            Sample_info sample_info = sampleScatteringDirectionProportionalToQMCMC(hit_point, ffnormal, prd.seed);

            new_direction = sample_info.direction;
            pdf = sample_info.pdf;
            if(dot(new_direction, ffnormal) <= 0.0){
                pdf = 0.0;
            }
        } else {
            // BRDF Sampling
            new_direction = Sample(mat, ffnormal, wo, prd.seed);
            pdf = Pdf(mat, ffnormal, wo, new_direction);
        }
    }
    pdf = max(pdf, 0.01);

    float3 f = Eval(mat, ffnormal, wo, new_direction);
    prd.t = t_hit;
    prd.origin = hit_point;
    prd.direction = new_direction;
    prd.scatterPdf = pdf;
	if (pdf > 0.0f){
	    prd.current_attenuation = f / pdf;
	}
	else
		prd.done = true;
}

RT_PROGRAM void any_hit()
{
	prd_shadow.inShadow = true;
	rtTerminateRay();
}

