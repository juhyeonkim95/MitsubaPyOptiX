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

using namespace optix;
// rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

namespace dielectric
{
RT_CALLABLE_PROGRAM BSDFSample3f Sample(MaterialParameter &mat, const float3 &normal, const float3 &wi, unsigned int &seed)
{
    BSDFSample3f bs;

    float ior = mat.intIOR / mat.extIOR;
    float eta = wi.z < 0.0f ? ior : 1 / ior;
    float cos_theta_t;
    const float F = fresnel::DielectricReflectance( eta, abs(wi.z), cos_theta_t );

	if( rnd(seed) <= F )
	{
		// Reflect
		bs.wo = make_float3(-wi.x, -wi.y, wi.z);
		bs.pdf = F;
		bs.weight = mat.albedo;
		bs.sampledLobe = BSDFLobe::SpecularReflectionLobe;
	}
	else
	{
	    // Refract
	    bs.wo = make_float3(-wi.x * eta, -wi.y * eta, -copysignf(cos_theta_t, wi.z));
	    bs.pdf = 1 - F;
	    bs.weight = mat.albedo * sqr(eta);
	    bs.sampledLobe = BSDFLobe::SpecularTransmissionLobe;
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
