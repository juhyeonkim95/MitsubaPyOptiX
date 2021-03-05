/*
 Copyright Disney Enterprises, Inc.  All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License
 and the following modification to it: Section 6 Trademarks.
 deleted and replaced with:

 6. Trademarks. This License does not grant permission to use the
 trade names, trademarks, service marks, or product names of the
 Licensor and its affiliates, except as required for reproducing
 the content of the NOTICE file.

 You may obtain a copy of the License at
 http://www.apache.org/licenses/LICENSE-2.0
*/

#include <optixu/optixu_math_namespace.h>
#include "optix/common/random.h"
#include "optix/common/sampling.h"
#include "optix/common/material_parameters.h"
#include "optix/bsdf/fresnel.h"
#include "optix/bsdf/bsdf_sample.h"
#include "optix/common/helpers.h"
#include "optix/bsdf/warp.h"

using namespace optix;

namespace plastic
{
RT_CALLABLE_PROGRAM BSDFSample3f Sample(MaterialParameter &mat, const float3 &normal, const float3 &wi, unsigned int &seed)
{
    BSDFSample3f bs;
    if(wi.z < 0){
        bs.pdf = 1.0;
        bs.weight = make_float3(0.0);
        return bs;
    }

    float ior = mat.intIOR / mat.extIOR;
    float eta = 1 / ior;
    float Fi = fresnel::DielectricReflectance( eta, wi.z);

    float s_mean = 1.0f;
    float d_mean = (mat.albedo.x + mat.albedo.x + mat.albedo.z) / 3.0f;
    float m_specular_sampling_weight = s_mean / (d_mean + s_mean);


    float prob_specular = Fi * m_specular_sampling_weight;
    float prob_diffuse = (1-Fi) * (1-m_specular_sampling_weight);

    prob_specular = prob_specular / (prob_specular + prob_diffuse);
    prob_diffuse = 1.f - prob_specular;

//    float substrateWeight = 1-Fi;
//    float specularWeight = Fi;
//    prob_specular = specularWeight / (specularWeight + substrateWeight);

    float m_fdr_int = fresnel::DiffuseFresnel(eta);

	if( rnd(seed) <= prob_specular )
	{
		// Reflect
		bs.wo = make_float3(-wi.x, -wi.y, wi.z);
		bs.pdf = prob_specular;
		bs.weight = make_float3(Fi / prob_specular);
		bs.sampledLobe = BSDFLobe::SpecularReflectionLobe;
	}
	else
	{
	    cosine_sample_hemisphere(rnd(seed), rnd(seed), bs.wo);
        float Fo = fresnel::DielectricReflectance(eta, bs.wo.z);

        float3 value = mat.albedo;
        // value = value / 1 - (value * m_fdr_int);
        value = value / (1 - (mat.nonlinear ? value * m_fdr_int : make_float3(m_fdr_int)));
        value *= ((1.0f - Fi)*(1.0f - Fo)*eta*eta);
        value = value / prob_diffuse;

        bs.pdf = warp::cosine_sample_hemisphere_pdf(bs.wo) * prob_diffuse;
        bs.weight = value;
        bs.sampledLobe = BSDFLobe::DiffuseReflectionLobe;
	}
    return bs;
}

RT_CALLABLE_PROGRAM float3 Eval(MaterialParameter &mat, float3 &normal, float3 &wi, float3 &wo)
{
    return make_float3(0.0f);
}

RT_CALLABLE_PROGRAM float Pdf(MaterialParameter &mat, float3 &normal, float3 &wi, float3 &wo){
    return 0.0f;
}
}
