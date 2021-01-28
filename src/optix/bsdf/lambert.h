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
#include "optix/common/prd_struct.h"

using namespace optix;
rtDeclareVariable(unsigned int,     sample_type, , );


RT_CALLABLE_PROGRAM float Pdf_lambert(float3 &ffnormal, float3 &wi)
{
    if(dot(ffnormal, wi) < 0.0)
        return 0.0f;
    float pdf;
    if(sample_type == 0){
        pdf = (1.0f/(2*M_PIf));
    } else {
        pdf = dot(ffnormal, wi) * (1.0f / M_PIf);
    }

	return pdf;
}

RT_CALLABLE_PROGRAM float3 Sample_lambert(float3 &ffnormal, float3 &wo, unsigned int &seed)
{
    float3 N = ffnormal;

	float3 dir;
    if(sample_type == 0){
        float z1=rnd(seed);
        float z2=rnd(seed);
        dir = UniformSampleHemisphere(z1, z2);
        optix::Onb onb( N );
        onb.inverse_transform( dir );

    } else {
        float r1 = rnd(seed);
        float r2 = rnd(seed);
        cosine_sample_hemisphere(r1, r2, dir);
        optix::Onb onb( N );
        onb.inverse_transform(dir);
    }

	return dir;
}


RT_CALLABLE_PROGRAM float3 Eval_lambert(MaterialParameter &mat, float3 &ffnormal, float3 &wo, float3 &wi)
{
	float3 N = ffnormal;
	float3 V = wo;
	float3 L = wi;

	float NDotL = dot(N, L);
	float NDotV = dot(N, V);
	if (NDotL <= 0.0f || NDotV <= 0.0f) return make_float3(0.0f);

	float3 out = (1.0f / M_PIf) * mat.diffuse_color;

	return out * clamp(dot(N, L), 0.0f, 1.0f);
}

//rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );
//
//RT_PROGRAM void dummy(){
//    float3 x = make_float3(0.0f);
//    PerRayData_pathtrace prd;
//    float a = Pdf(x, x);
//    float3 b = make_float3(a, a, a);
//    Sample(b, b, prd);
//    current_prd.direction = Eval(x, x, x, x);
//    current_prd.origin = make_float3(a);
//}