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

using namespace optix;
namespace conductor
{
RT_CALLABLE_PROGRAM BSDFSample3f Sample(MaterialParameter &mat, const float3 &normal, const float3 &wi, unsigned int &seed)
{
    BSDFSample3f bs;
    if(wi.z < 0){
        bs.pdf = 1.0;
        bs.weight = make_float3(0.0);
        return bs;
    }
    bs.wo = make_float3(-wi.x, -wi.y, wi.z);
    bs.pdf = 1.0f;
    bs.weight = mat.albedo * fresnel::ConductorReflectance(mat.eta, mat.k, wi.z);
    bs.sampledLobe = BSDFLobe::SpecularReflectionLobe;
    return bs;
}

RT_CALLABLE_PROGRAM float3 Eval(MaterialParameter &mat, const float3 &normal, const float3 &wi, const float3 &wo)
{
    return make_float3(0.0f);
}

RT_CALLABLE_PROGRAM float Pdf(MaterialParameter &mat, const float3 &normal, const float3 &wi, const float3 &wo)
{
    return 1.0f;
}


}
