#include "optix/bsdf/lambert.h"
using namespace optix;
rtDeclareVariable(int, programId, , );

// Evaluate pdf for wo and wi
RT_CALLABLE_PROGRAM float Pdf(MaterialParameter &mat, float3 &ffnormal, float3 &wo, float3 &wi)
{
	if(programId == 0){
	    return Pdf_lambert(ffnormal, wi);
	}
	return 0;
}

// Sample for wo using brdf
RT_CALLABLE_PROGRAM float3 Sample(MaterialParameter &mat, float3 &ffnormal, float3 &wo, unsigned int &seed)
{
    if(programId == 0){
	    return Sample_lambert(ffnormal, wo, seed);
	}
	return make_float3(0.0f);
}

// Evaluate brdf * cos value for wo and wi.
RT_CALLABLE_PROGRAM float3 Eval(MaterialParameter &mat, float3 &ffnormal, float3 &wo, float3 &wi)
{
    if(programId == 0){
	    return Eval_lambert(mat, ffnormal, wo, wi);
	}
	return make_float3(0.0f);
}