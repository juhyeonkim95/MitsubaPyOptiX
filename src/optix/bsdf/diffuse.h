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
#include "optix/bsdf/bsdf_sample.h"
#include "optix/bsdf/warp.h"
#include "optix/utils/material_value_loader.h"


using namespace optix;
namespace diffuse
{
__device__ uint32_t flags = BSDFFlags::DiffuseReflection | BSDFFlags::FrontSide;

RT_CALLABLE_PROGRAM void Sample(
    const MaterialParameter &mat, const SurfaceInteraction &si,
    unsigned int &seed, BSDFSample3f &bs
) {
    float3 diffuse_reflectance = eval_diffuse_reflectance(mat, si);

    if(si.wi.z < 0){
        bs.pdf = 0.0;
        bs.weight = make_float3(0.0);
        return;
    }
    cosine_sample_hemisphere(rnd(seed), rnd(seed), bs.wo);
    bs.pdf = warp::cosine_sample_hemisphere_pdf(bs.wo);
    bs.weight = diffuse_reflectance;
    bs.sampledLobe = BSDFLobe::DiffuseReflectionLobe;
	return;
}

RT_CALLABLE_PROGRAM float3 Eval(const MaterialParameter &mat, const SurfaceInteraction &si, const float3 &wo)
{
    float3 diffuse_reflectance = eval_diffuse_reflectance(mat, si);
    if(si.wi.z <= 0.0f || wo.z <= 0.0f)
        return make_float3(0.0f);
    return diffuse_reflectance * M_1_PIf * wo.z;
}

RT_CALLABLE_PROGRAM float Pdf(const MaterialParameter &mat, const SurfaceInteraction &si, const float3 &wo)
{
    if(si.wi.z <= 0.0f || wo.z <= 0.0f)
        return 0.0f;
    return warp::cosine_sample_hemisphere_pdf(wo);
}

}