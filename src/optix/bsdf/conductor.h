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
#include "optix/utils/material_value_loader.h"

using namespace optix;
namespace conductor
{
__device__ uint32_t flags = BSDFFlags::DeltaReflection | BSDFFlags::FrontSide;

RT_CALLABLE_PROGRAM void Sample(
    const MaterialParameter &mat, const SurfaceInteraction &si,
    unsigned int &seed, BSDFSample3f &bs
)
{
    float3 specular_reflectance = eval_specular_reflectance(mat, si);
    if(si.wi.z < 0){
        bs.pdf = 1.0;
        bs.weight = make_float3(0.0);
        return;
    }
    bs.wo = make_float3(-si.wi.x, -si.wi.y, si.wi.z);
    bs.pdf = 1.0f;
    bs.weight = specular_reflectance * fresnel::ConductorReflectance(mat.eta, mat.k, si.wi.z);
    bs.sampledLobe = BSDFLobe::SpecularReflectionLobe;
    return;
}

RT_CALLABLE_PROGRAM float3 Eval(const MaterialParameter &mat, const SurfaceInteraction &si, const float3 &wo)
{
    return make_float3(0.0f);
}

RT_CALLABLE_PROGRAM float Pdf(const MaterialParameter &mat, const SurfaceInteraction &si, const float3 &wo)
{
    return 1.0f;
}


}
