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
#include "optix/common/helpers.h"

using namespace optix;
namespace roughdielectric
{

RT_CALLABLE_PROGRAM BSDFSample3f SampleBase(MaterialParameter &mat, bool sampleR, bool sampleT,
    const float3 &normal, const float3 &wi, unsigned int &seed)
{
    BSDFSample3f bs;
    bs.pdf = 1.0;
    bs.weight = make_float3(0.0);

    float wiDotN = wi.z;
    float ior = mat.intIOR / mat.extIOR;
    float eta = wi.z < 0.0f ? ior : 1 / ior;

    float r1 = rnd(seed);
    float r2 = rnd(seed);
    DistributionEnum dist = static_cast<DistributionEnum>(mat.distribution_type);

    float sampleRoughness = (1.2f - 0.2f*sqrtf(abs(wiDotN)))*mat.roughness;
    float alpha = microfacet::roughnessToAlpha(dist, mat.roughness);
    float sampleAlpha  = microfacet::roughnessToAlpha(dist, sampleRoughness);

    float3 m = microfacet::sample(dist, sampleAlpha , r1, r2);
    float pm = microfacet::pdf(dist, sampleAlpha , m);

    float wiDotM = dot(wi, m);
    float cosThetaT;
    const float F = fresnel::DielectricReflectance( 1.0f / ior , wiDotM, cosThetaT );
    float etaM = wiDotM < 0.0f ? ior : 1.0f/ior;

    float3 wo;
    bool reflect;
    if (sampleR && sampleT){
        reflect = rnd(seed) < F;
    } else if (sampleT){
        reflect = false;
    } else if (sampleR){
        reflect = true;
    } else{
        return bs;
    }

    if (reflect)
    {
        // wo = optix::reflect(-wi, m);
        wo = 2.0f*wiDotM*m - wi;
    }
    else // Transmission
    {
	    //optix::refract( wo, -wi, m, 1 / etaM );
	    wo = (etaM*wiDotM - (wiDotM > 0? 1: -1)*cosThetaT)*m - etaM*wi;
    }

    float woDotN = wo.z;
    bool reflected = wiDotN*woDotN > 0.0f;
    if (reflected != reflect)
        return bs;

    float woDotM = dot(wo, m);
    float G = microfacet::G(dist, alpha, wi, wo, m);
    float D = microfacet::D(dist, alpha, m);
    bs.weight = mat.albedo*abs(wiDotM)*G*D/(abs(wiDotN) * pm);
    bs.wo = wo;

    if(reflect){
        bs.pdf = pm * 0.25f / abs(wiDotM);
        bs.sampledLobe = BSDFLobe::GlossyReflectionLobe;
        //bs.pdf *= F;
    } else {
        float denom = (eta*wiDotM + woDotM);
        bs.pdf = pm * abs(woDotM) / (denom * denom);
        //bs.pdf *= (1 - F);
        bs.weight *= sqr(eta);
        bs.sampledLobe = BSDFLobe::GlossyTransmissionLobe;
    }

    if (sampleR && sampleT) {
        if (reflect)
            bs.pdf *= F;
        else
            bs.pdf *= 1.0f - F;
    } else {
        if (reflect)
            bs.weight *= F;
        else
            bs.weight *= 1.0f - F;
    }

    return bs;
}

RT_CALLABLE_PROGRAM BSDFSample3f Sample(MaterialParameter &mat,
    const float3 &normal, const float3 &wi, unsigned int &seed)
{
    return SampleBase(mat, true, true, normal, wi, seed);
}

RT_CALLABLE_PROGRAM float3 Eval(MaterialParameter &mat, const float3 &normal, const float3 &wi, const float3 &wo)
{
    float ior = mat.intIOR / mat.extIOR;
    float alpha  = mat.roughness;
    DistributionEnum dist = static_cast<DistributionEnum>(mat.distribution_type);

    float wiDotN = wi.z;
    float woDotN = wo.z;
    bool reflect = wiDotN*woDotN >= 0.0f;

    float eta = wiDotN < 0.0f? ior : 1.0f/ior;
    float3 m;
    if(reflect)
        m = sgnE(wiDotN)*normalize(wi + wo);
    else
        m = -normalize(wi*eta + wo);

    float wiDotM = dot(wi, m);
    float woDotM = dot(wo, m);
    float F = fresnel::DielectricReflectance(1.0f/ior, wiDotM);
    float G = microfacet::G(dist, alpha, wi, wo, m);
    float D = microfacet::D(dist, alpha, m);

    if (reflect) {
        float fr = (F*G*D*0.25f)/abs(wiDotN);
        return mat.albedo * fr;
    } else {
        float fs = abs(wiDotM*woDotM)*(1.0f - F)*G*D/(sqr(eta*wiDotM + woDotM)*abs(wiDotN));
        return mat.albedo * fs;
    }
}

RT_CALLABLE_PROGRAM float PdfBase(MaterialParameter &mat, bool sampleR, bool sampleT,
const float3 &normal, const float3 &wi, const float3 &wo){
    float ior = mat.intIOR / mat.extIOR;
    DistributionEnum dist = static_cast<DistributionEnum>(mat.distribution_type);
    float wiDotN = wi.z;
    float woDotN = wo.z;
    bool reflect = wiDotN*woDotN >= 0.0f;

    float sampleRoughness = (1.2f - 0.2f*sqrtf(abs(wiDotN)))*mat.roughness;
    float sampleAlpha  = microfacet::roughnessToAlpha(dist, sampleRoughness);


    float eta = wiDotN < 0.0f? ior : 1.0f/ior;
    float3 m;
    if(reflect)
        m = sgnE(wiDotN)*normalize(wi + wo);
    else
        m = -normalize(wi*eta + wo);

    float wiDotM = dot(wi, m);
    float woDotM = dot(wo, m);
    float F = fresnel::DielectricReflectance(1.0f/ior, wiDotM);
    float pm = microfacet::pdf(dist, sampleAlpha, m);

    float pdf;
    if (reflect) {
        pdf = pm*0.25f/abs(wiDotM);
    } else {
        pdf = pm*abs(woDotM)/sqr(eta*wiDotM + woDotM);
    }
    if (sampleR && sampleT) {
        if (reflect)
            pdf *= F;
        else
            pdf *= 1.0f - F;
    }
    return pdf;
}

RT_CALLABLE_PROGRAM float Pdf(MaterialParameter &mat, const float3 &normal, const float3 &wi, const float3 &wo)
{
    return PdfBase(mat, true, true, normal, wi, wo);
}
}
