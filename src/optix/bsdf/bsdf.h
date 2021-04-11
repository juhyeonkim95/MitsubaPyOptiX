#pragma once

#include "optix/bsdf/lambert.h"
#include "optix/bsdf/dielectric.h"
#include "optix/bsdf/roughdielectric.h"
#include "optix/bsdf/conductor.h"
#include "optix/bsdf/roughconductor.h"
#include "optix/bsdf/plastic.h"
#include "optix/bsdf/roughplastic.h"

//#include "optix/bsdf/disney2.h"


using namespace optix;
rtDeclareVariable(int, programId, , );

// Sample for wo using brdf
RT_CALLABLE_PROGRAM BSDFSample3f Sample(const MaterialParameter &mat, const float3 &wi, unsigned int &seed)
{
    switch(programId){
    case 0: return lambert::Sample(mat, wi, seed);
    case 1: return dielectric::Sample(mat, wi, seed);
    case 2: return roughdielectric::Sample(mat, wi, seed);
    case 3: return conductor::Sample(mat, wi, seed);
    case 4: return roughconductor::Sample(mat, wi, seed);
    case 5: return plastic::Sample(mat, wi, seed);
    case 6: return roughplastic::Sample(mat, wi, seed);
    }
	BSDFSample3f bs;
	return bs;
}


// Evaluate pdf for wo and wi
RT_CALLABLE_PROGRAM float Pdf(const MaterialParameter &mat, const float3 &wi, const float3 &wo)
{
    switch(programId){
    case 0:return lambert::Pdf(mat, wi, wo);
    case 1:return dielectric::Pdf(mat, wi, wo);
    case 2:return roughdielectric::Pdf(mat, wi, wo);
    case 3:return conductor::Pdf(mat, wi, wo);
    case 4:return roughconductor::Pdf(mat, wi, wo);
    case 5:return plastic::Pdf(mat, wi, wo);
    case 6:return roughplastic::Pdf(mat, wi, wo);
    default: return 0;
    }
}

// Evaluate brdf * cos value for wo and wi.
RT_CALLABLE_PROGRAM float3 Eval(const MaterialParameter &mat, const float3 &wi, const float3 &wo)
{
    switch(programId){
    case 0: return lambert::Eval(mat, wi, wo);
    case 1: return dielectric::Eval(mat, wi, wo);
    case 2: return roughdielectric::Eval(mat, wi, wo);
    case 3: return conductor::Eval(mat, wi, wo);
    case 4: return roughconductor::Eval(mat, wi, wo);
    case 5: return plastic::Eval(mat, wi, wo);
    case 6: return roughplastic::Eval(mat, wi, wo);
    default: return make_float3(0.0f);
    }
}