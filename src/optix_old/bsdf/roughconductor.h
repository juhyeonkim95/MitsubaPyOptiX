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
#include "optix/bsdf/material_constants.h"
#include "optix/bsdf/microfacet.h"
#include "optix/bsdf/bsdf_sample.h"

using namespace optix;
namespace roughconductor
{

RT_CALLABLE_PROGRAM BSDFSample3f Sample(MaterialParameter &mat, const float3 &normal, const float3 &wi, unsigned int &seed)
{
    BSDFSample3f bs;
    bs.pdf = 1.0;
    bs.weight = make_float3(0.0);
    if(wi.z < 0){
        return bs;
    }

    float r1 = rnd(seed);
    float r2 = rnd(seed);
    DistributionEnum dist = static_cast<DistributionEnum>(mat.distribution_type);

    // float sampleRoughness = (1.2f - 0.2f*sqrtf(abs(wiDotN)))*mat.roughness;
    float sampleRoughness = mat.roughness;
    float alpha = microfacet::roughnessToAlpha(dist, mat.roughness);
    float sampleAlpha  = microfacet::roughnessToAlpha(dist, sampleRoughness);

    float3 m = microfacet::sample(dist, sampleAlpha, r1, r2);
    float wiDotM = dot(wi, m);
    float3 wo = 2.0f*wiDotM*m - wi;
    if (wiDotM <= 0.0f || wo.z <= 0.0f)
        return bs;

    float G = microfacet::G(dist, alpha, wi, bs.wo, m);
    float D = microfacet::D(dist, alpha, m);
    float mPdf = microfacet::pdf(dist, sampleAlpha, m);
    float pdf = mPdf * 0.25f / wiDotM;
    float weight = wiDotM * G * D / (wi.z * mPdf);
    float3 F = fresnel::ConductorReflectance(eta, k, wiDotM);

    bs.wo = wo;
    bs.pdf = pdf;
    bs.weight = mat.albedo * F * weight;
    bs.sampledLobe = BSDFLobe::GlossyReflectionLobe;

    return bs;
}

RT_CALLABLE_PROGRAM float3 Eval(MaterialParameter &mat, const float3 &normal, const float3 &wi, const float3 &wo)
{
    if(wi.z <= 0.0f || wo.z <= 0.0f){
        return make_float3(0.0f);
    }


    DistributionEnum dist = static_cast<DistributionEnum>(mat.distribution_type);
    float alpha = microfacet::roughnessToAlpha(dist, mat.roughness);

    float3 hr = normalize(wo + wi);
    float cosThetaM = dot(wi, hr);
    float3 F = fresnel::ConductorReflectance(eta, k, cosThetaM);
    float G = microfacet::G(dist, alpha, wi, wo, hr);
    float D = microfacet::D(dist, alpha, hr);
    float fr = (G*D*0.25f)/wi.z;

    return mat.albedo * F * fr;
}

RT_CALLABLE_PROGRAM float Pdf(MaterialParameter &mat, float3 &normal, float3 &wo, float3 &wi)
{
    if(wi.z <= 0.0f || wo.z <= 0.0f){
        return 0.0f;
    }
    DistributionEnum dist = static_cast<DistributionEnum>(mat.distribution_type);

    // float sampleRoughness = (1.2f - 0.2f*sqrtf(abs(wiDotN)))*mat.roughness;
    float sampleRoughness = mat.roughness;
    float sampleAlpha  = microfacet::roughnessToAlpha(dist, sampleRoughness);

    float3 hr = normalize(wo + wi);
    float mPdf = microfacet::pdf(dist, sampleAlpha, hr);
    float pdf = mPdf * 0.25f / dot(wi, hr);
    return pdf;
}
}
