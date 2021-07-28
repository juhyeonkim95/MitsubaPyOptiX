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
#pragma once

#include <optixu/optixu_math_namespace.h>
#include "optix/common/random.h"
#include "optix/common/sampling.h"
#include "optix/common/material_parameters.h"
#include "optix/bsdf/fresnel.h"
#include "optix/bsdf/bsdf_sample.h"
#include "optix/common/helpers.h"

using namespace optix;
// rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
// rtDeclareVariable(bool, m_nonlinear, , );

namespace roughplastic
{
__device__ uint32_t flags = BSDFFlags::GlossyReflection | BSDFFlags::DiffuseReflection | BSDFFlags::FrontSide;

RT_CALLABLE_PROGRAM void Sample(
    const MaterialParameter &mat, const SurfaceInteraction &si,
    unsigned int &seed, BSDFSample3f &bs
)
{
    const float3 &wi = si.wi;

    if(wi.z < 0){
        bs.pdf = 1.0;
        bs.weight = make_float3(0.0);
        return;
    }

    float3 diffuse_reflectance = eval_diffuse_reflectance(mat, si);
    float ior = mat.intIOR / mat.extIOR;
    float eta = 1 / ior;
    float Fi = fresnel::DielectricReflectance(eta, wi.z);

    float s_mean = 1.0f;
    float d_mean = (diffuse_reflectance.x + diffuse_reflectance.x + diffuse_reflectance.z) / 3.0f;
    float m_specular_sampling_weight = s_mean / (d_mean + s_mean);


    float prob_specular = Fi * m_specular_sampling_weight;
    float prob_diffuse = (1-Fi) * (1-m_specular_sampling_weight);

    prob_specular = prob_specular / (prob_specular + prob_diffuse);
    prob_diffuse = 1.f - prob_specular;
    float specularProbability = prob_specular;
//    float substrateWeight = 1-Fi;
//    float specularWeight = Fi;
//    prob_specular = specularWeight / (specularWeight + substrateWeight);

    float _diffuseFresnel = fresnel::DiffuseFresnel(eta);

	if( rnd(seed) <= prob_specular )
	{
        roughdielectric::SampleBase(mat, true, false, si, seed, bs);

        float3 diffuseAlbedo = diffuse_reflectance;
        float Fo = fresnel::DielectricReflectance(eta, bs.wo.z);

        float3 temp = (mat.nonlinear ? diffuseAlbedo * _diffuseFresnel : make_float3(_diffuseFresnel));

        float3 brdfSubstrate = ((1.0f - Fi)*(1.0f - Fo)*eta*eta)
                *(diffuseAlbedo/(1.0f - temp))*M_1_PIf*bs.wo.z;
        float3 brdfSpecular = bs.weight*bs.pdf;
        float pdfSubstrate = warp::cosine_sample_hemisphere_pdf(bs.wo)*(1.0f - specularProbability);
        float pdfSpecular = bs.pdf*specularProbability;

        bs.weight = (brdfSpecular + brdfSubstrate)/(pdfSpecular + pdfSubstrate);
        bs.pdf = pdfSpecular + pdfSubstrate;
        return;
	}
	else
	{
	    cosine_sample_hemisphere(rnd(seed), rnd(seed), bs.wo);
        float Fo = fresnel::DielectricReflectance(eta, bs.wo.z);
        float3 diffuseAlbedo = diffuse_reflectance;

        float3 temp = (mat.nonlinear ? diffuseAlbedo * _diffuseFresnel : make_float3(_diffuseFresnel));
        bs.weight = ((1.0f - Fi)*(1.0f - Fo)*eta*eta)*(diffuseAlbedo/(1.0f - temp));
        bs.pdf = warp::cosine_sample_hemisphere_pdf(bs.wo);

        float3 brdfSubstrate = bs.weight*bs.pdf;
        float  pdfSubstrate = bs.pdf*(1.0f - specularProbability);
        float3 brdfSpecular = roughdielectric::Eval(mat, si, bs.wo);
        float pdfSpecular  = roughdielectric::PdfBase(mat, true, false, si, bs.wo);
        pdfSpecular *= specularProbability;

        bs.weight = (brdfSpecular + brdfSubstrate)/(pdfSpecular + pdfSubstrate);
        bs.pdf = pdfSpecular + pdfSubstrate;
	}
    return;
}

RT_CALLABLE_PROGRAM float3 Eval(const MaterialParameter &mat, const SurfaceInteraction &si, const float3 &wo)
{
    return make_float3(0.0f);
}

RT_CALLABLE_PROGRAM float Pdf(const MaterialParameter &mat, const SurfaceInteraction &si, const float3 &wo){
    return 0.0f;
}
}
