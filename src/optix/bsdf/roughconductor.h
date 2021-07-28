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
#include "optix/bsdf/microfacet.h"
#include "optix/bsdf/bsdf_sample.h"
#include "optix/utils/material_value_loader.h"

using namespace optix;
namespace roughconductor
{
__device__ uint32_t flags = BSDFFlags::GlossyReflection | BSDFFlags::FrontSide;

RT_CALLABLE_PROGRAM void Sample(
    const MaterialParameter &mat, const SurfaceInteraction &si,
    unsigned int &seed, BSDFSample3f &bs
)
{
    const float3 &wi = si.wi;

    float3 specular_reflectance = eval_specular_reflectance(mat, si);
    float roughness = eval_roughness(mat, si);

    bs.pdf = 1.0;
    bs.weight = make_float3(0.0);
    if(wi.z < 0){
        return;
    }

    float r1 = rnd(seed);
    float r2 = rnd(seed);
    DistributionEnum dist = static_cast<DistributionEnum>(mat.distribution_type);

    // float sampleRoughness = (1.2f - 0.2f*sqrtf(abs(wiDotN)))*mat.roughness;
    float sampleRoughness = roughness;
    float alpha = microfacet::roughnessToAlpha(dist, roughness);
    float sampleAlpha  = microfacet::roughnessToAlpha(dist, sampleRoughness);

    float3 m = microfacet::sample(dist, sampleAlpha, r1, r2);
    float wiDotM = dot(wi, m);
    float3 wo = 2.0f*wiDotM*m - wi;
    if (wiDotM <= 0.0f || wo.z <= 0.0f)
        return;

    float G = microfacet::G(dist, alpha, wi, bs.wo, m);
    float D = microfacet::D(dist, alpha, m);
    float mPdf = microfacet::pdf(dist, sampleAlpha, m);
    float pdf = mPdf * 0.25f / wiDotM;
    float weight = wiDotM * G * D / (wi.z * mPdf);
    float3 F = fresnel::ConductorReflectance(mat.eta, mat.k, wiDotM);

    bs.wo = wo;
    bs.pdf = pdf;
    bs.weight = specular_reflectance * F * weight;
    bs.sampledLobe = BSDFLobe::GlossyReflectionLobe;

    return;
}

RT_CALLABLE_PROGRAM float3 Eval(const MaterialParameter &mat, const SurfaceInteraction &si, const float3 &wo)
{
    const float3 &wi = si.wi;

    if(wi.z <= 0.0f || wo.z <= 0.0f){
        return make_float3(0.0f);
    }

    float roughness = eval_roughness(mat, si);
    float3 specular_reflectance = eval_specular_reflectance(mat, si);

    DistributionEnum dist = static_cast<DistributionEnum>(mat.distribution_type);
    float alpha = microfacet::roughnessToAlpha(dist, roughness);

    float3 hr = normalize(wo + wi);
    float cosThetaM = dot(wi, hr);
    float3 F = fresnel::ConductorReflectance(mat.eta, mat.k, cosThetaM);
    float G = microfacet::G(dist, alpha, wi, wo, hr);
    float D = microfacet::D(dist, alpha, hr);
    float fr = (G*D*0.25f)/wi.z;

    return specular_reflectance * F * fr;
}

RT_CALLABLE_PROGRAM float Pdf(const MaterialParameter &mat, const SurfaceInteraction &si, const float3 &wo)
{
    const float3 &wi = si.wi;

    if(wi.z <= 0.0f || wo.z <= 0.0f){
        return 0.0f;
    }
    DistributionEnum dist = static_cast<DistributionEnum>(mat.distribution_type);

    // float sampleRoughness = (1.2f - 0.2f*sqrtf(abs(wiDotN)))*mat.roughness;
    float sampleRoughness = eval_roughness(mat, si);
    // float sampleRoughness = mat.roughness;
    float sampleAlpha  = microfacet::roughnessToAlpha(dist, sampleRoughness);

    float3 hr = normalize(wo + wi);
    float mPdf = microfacet::pdf(dist, sampleAlpha, hr);
    float pdf = mPdf * 0.25f / dot(wi, hr);
    return pdf;
}
}
