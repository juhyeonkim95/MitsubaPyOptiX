#include <optixu/optixu_math_namespace.h>
#include "optix/common/prd_struct.h"
#include "optix/common/random.h"
#include "optix/common/prd_struct.h"
#include "optix/common/material_parameters.h"
using namespace optix;

rtDeclareVariable(PerRayData_pathtrace_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(PerRayData_pathtrace, prd, rtPayload, );

rtDeclareVariable(int, materialId, , );
rtBuffer<MaterialParameter> sysMaterialParameters;
rtDeclareVariable( float3, texcoord, attribute texcoord, );


RT_PROGRAM void any_hit_shadow()
{
	prd_shadow.inShadow = true;
	rtTerminateRay();
}


RT_PROGRAM void any_hit_cutout()
{
	float opacity = 1.0f;
	const int id = sysMaterialParameters[materialId].albedoID;
	if (id != RT_TEXTURE_ID_NULL){
	    opacity = optix::rtTex2D<float4>(id , texcoord.x, 1 - texcoord.y).w;
	} else {
	    opacity = sysMaterialParameters[materialId].opacity;
	}

	if(opacity < 1.0f && opacity <= rnd(prd.seed)){
	    rtIgnoreIntersection();
	}
}

RT_PROGRAM void any_hit_shadow_cutout()
{
	float opacity = 1.0f;
	const int id = sysMaterialParameters[materialId].albedoID;
	if (id != RT_TEXTURE_ID_NULL){
	    opacity = optix::rtTex2D<float4>(id , texcoord.x, 1 - texcoord.y).w;
	} else {
	    opacity = sysMaterialParameters[materialId].opacity;
	}

	if(opacity < 1.0f && opacity <= rnd(prd.seed)){
	    rtIgnoreIntersection();
	} else {
	    prd_shadow.inShadow = true;
	    rtTerminateRay();
	}
}