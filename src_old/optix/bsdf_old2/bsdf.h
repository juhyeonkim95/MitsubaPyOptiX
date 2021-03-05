#pragma once

#include "optix/bsdf/lambert.h"
#include "optix/bsdf/glass.h"
#include "optix/bsdf/disney2.h"

using namespace optix;
rtDeclareVariable(int, programId, , );

// Evaluate pdf for wo and wi
RT_CALLABLE_PROGRAM float Pdf(MaterialParameter &mat, float3 &normal, float3 &wo, float3 &wi)
{
	if(programId == 0){
	    return lambert::Pdf(normal, wi);
	} else if (programId == 1){
	    return glass::Pdf(normal, wi);
	} else if (programId == 2){
        return disney_old::Pdf(mat, normal, wo, wi);
	}
	return 0;
}

// Sample for wo using brdf
RT_CALLABLE_PROGRAM float3 Sample(MaterialParameter &mat, float3 &normal, float3 &wo, unsigned int &seed)
{
    if(programId == 0){
	    return lambert::Sample(normal, wo, seed);
	} else if (programId == 1){
	    return glass::Sample(mat, normal, wo, seed);
	} else if (programId == 2){
	    return disney_old::Sample(mat, normal, wo, seed);
	}
	return make_float3(0.0f);
}

// Evaluate brdf * cos value for wo and wi.
RT_CALLABLE_PROGRAM float3 Eval(MaterialParameter &mat, float3 &normal, float3 &wo, float3 &wi)
{
    if(programId == 0){
	    return lambert::Eval(mat, normal, wo, wi);
	} else if(programId == 1){
	    return glass::Eval(mat, normal, wo, wi);
	} else if (programId == 2){
	    return disney_old::Eval(mat, normal, wo, wi);
	}
	return make_float3(0.0f);
}