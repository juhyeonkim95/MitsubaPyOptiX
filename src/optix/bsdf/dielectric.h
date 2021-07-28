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
#include "optix/utils/material_value_loader.h"

using namespace optix;

namespace dielectric
{
__device__ uint32_t flags = BSDFFlags::DeltaReflection | BSDFFlags::DeltaTransmission | BSDFFlags::FrontSide | BSDFFlags::BackSide | BSDFFlags::NonSymmetric;

RT_CALLABLE_PROGRAM void Sample(
    const MaterialParameter &mat, const SurfaceInteraction &si,
    unsigned int &seed, BSDFSample3f &bs
)
{

    float ior = mat.intIOR / mat.extIOR;
    float eta = si.wi.z < 0.0f ? ior : 1 / ior;
    float cos_theta_t;
    const float F = fresnel::DielectricReflectance( eta, abs(si.wi.z), cos_theta_t );

	if( rnd(seed) <= F )
	{
		// Reflect
		bs.wo = make_float3(-si.wi.x, -si.wi.y, si.wi.z);
		bs.pdf = F;
		bs.weight = eval_specular_reflectance(mat, si);
		bs.sampledLobe = BSDFLobe::SpecularReflectionLobe;
	}
	else
	{
	    // Refract
	    bs.wo = make_float3(-si.wi.x * eta, -si.wi.y * eta, -copysignf(cos_theta_t, si.wi.z));
	    bs.pdf = 1 - F;
	    bs.weight = eval_specular_transmittance(mat, si) * sqr(eta);
	    bs.sampledLobe = BSDFLobe::SpecularTransmissionLobe;
	}
    return;
}

RT_CALLABLE_PROGRAM float3 Eval(const MaterialParameter &mat, const SurfaceInteraction &si, const float3 &wo)
{
    return make_float3(0.0f);
}

RT_CALLABLE_PROGRAM float Pdf(const MaterialParameter &mat, const SurfaceInteraction &si, const float3 &wo)
{
    return 0.0f;
}
}
