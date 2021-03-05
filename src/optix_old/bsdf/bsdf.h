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

// Evaluate pdf for wo and wi
RT_CALLABLE_PROGRAM float Pdf(MaterialParameter &mat, float3 &normal, float3 &wi, float3 &wo)
{
	if(programId == 0){
	    return lambert::Pdf(mat, normal, wi, wo);
	} else if (programId == 1){
	    return dielectric::Pdf(mat, normal, wi, wo);
	} else if (programId == 2){
        return roughdielectric::Pdf(mat, normal, wi, wo);
	} else if (programId == 3){
        return conductor::Pdf(mat, normal, wi, wo);
	} else if (programId == 4){
        return roughconductor::Pdf(mat, normal, wi, wo);
	} else if (programId == 5){
        return plastic::Pdf(mat, normal, wi, wo);
	} else if (programId == 6){
        return roughplastic::Pdf(mat, normal, wi, wo);
	}
	return 0;
}

// Sample for wo using brdf
RT_CALLABLE_PROGRAM BSDFSample3f Sample(MaterialParameter &mat, float3 &normal, float3 &wi, unsigned int &seed)
{
    if(programId == 0){
	    return lambert::Sample(mat, normal, wi, seed);
	} else if (programId == 1){
	    return dielectric::Sample(mat, normal, wi, seed);
	} else if (programId == 2){
	    return roughdielectric::Sample(mat, normal, wi, seed);
	} else if (programId == 3){
        return conductor::Sample(mat, normal, wi, seed);
	} else if (programId == 4){
        return roughconductor::Sample(mat, normal, wi, seed);
	} else if (programId == 5){
        return plastic::Sample(mat, normal, wi, seed);
	} else if (programId == 6){
        return roughplastic::Sample(mat, normal, wi, seed);
	}
	BSDFSample3f bs;
	return bs;
}

// Evaluate brdf * cos value for wo and wi.
RT_CALLABLE_PROGRAM float3 Eval(MaterialParameter &mat, float3 &normal, float3 &wi, float3 &wo)
{
    if(programId == 0){
	    return lambert::Eval(mat, normal, wi, wo);
	} else if(programId == 1){
	    return dielectric::Eval(mat, normal, wi, wo);
	} else if (programId == 2){
	    return roughdielectric::Eval(mat, normal, wi, wo);
	} else if (programId == 3){
        return conductor::Eval(mat, normal, wi, wo);
	} else if (programId == 4){
        return roughconductor::Eval(mat, normal, wi, wo);
	} else if (programId == 5){
        return plastic::Eval(mat, normal, wi, wo);
	} else if (programId == 6){
        return roughplastic::Eval(mat, normal, wi, wo);
	}
	return make_float3(0.0f);
}