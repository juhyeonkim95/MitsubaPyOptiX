#include <optixu/optixu_math_namespace.h>
#include "optix/common/prd_struct.h"
#include "optix/common/random.h"
#include "optix/common/prd_struct.h"
#include "optix/common/material_parameters.h"
#include "optix/texture/texture.h"
#include "optix/utils/material_value_loader.h"

using namespace optix;

rtDeclareVariable(PerRayData_pathtrace_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(SurfaceInteraction, si, rtPayload, );
rtDeclareVariable( float3, texcoord, attribute texcoord, );

// Material parameter definition.
rtDeclareVariable(int, materialId, , );
rtBuffer<MaterialParameter> sysMaterialParameters;

// The shadow ray program for all materials with no cutout opacity.
RT_PROGRAM void any_hit_shadow()
{
	prd_shadow.inShadow = true;
	rtTerminateRay();
}

// One anyhit program for the radiance ray for all materials with cutout opacity!
RT_PROGRAM void any_hit_cutout()
{
	float opacity = eval_opacity(sysMaterialParameters[materialId], texcoord);
	// Stochastic alpha test to get an alpha blend effect.
	if(opacity < 1.0f && opacity <= rnd(si.seed)){
	    rtIgnoreIntersection();
	}
}

// For the shadow ray type.
RT_PROGRAM void any_hit_shadow_cutout()
{
    float opacity = eval_opacity(sysMaterialParameters[materialId], texcoord);
    // Stochastic alpha test to get an alpha blend effect.
	if(opacity < 1.0f && opacity <= rnd(prd_shadow.seed)){
	    rtIgnoreIntersection();
	} else {
	    prd_shadow.inShadow = true;
	    rtTerminateRay();
	}
}
