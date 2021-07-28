#pragma once

#include <optixu/optixu_math_namespace.h>
#include "optix/common/random.h"
#include "optix/common/sampling.h"
#include "optix/common/material_parameters.h"
#include "optix/bsdf/bsdf_sample.h"
#include "optix/bsdf/warp.h"

using namespace optix;
rtDeclareVariable(unsigned int,  texture_id, , );
rtDeclareVariable(unsigned int,  texture_id, , );
rtDeclareVariable(unsigned int,  texture_id, , );
namespace bitmap
{

RT_FUNCTION float3 evaluate(const float3& texcoord){
    const float3 texColor = make_float3(optix::rtTex2D<float4>(texture_id , texcoord.x, 1 - texcoord.y));
    return make_float3(powf(texColor.x, 2.2f), powf(texColor.y, 2.2f), powf(texColor.z, 2.2f)); // to linear space
}
}