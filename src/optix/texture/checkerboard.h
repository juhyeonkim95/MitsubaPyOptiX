#pragma once

#include <optixu/optixu_math_namespace.h>
#include "optix/common/random.h"
#include "optix/common/sampling.h"
#include "optix/common/material_parameters.h"
#include "optix/bsdf/bsdf_sample.h"
#include "optix/bsdf/warp.h"

using namespace optix;
rtDeclareVariable(float3, color0, , );
rtDeclareVariable(float3, color1, , );
rtDeclareVariable(Matrix3x3,  to_uv, , );

namespace checkerboard
{

RT_FUNCTION float3 evaluate(const float3& texcoord){
    float3 texcoordTransformed = to_uv * texcoord;
    float u = texcoordTransformed.x - int(texcoordTransformed.x);
    float v = texcoordTransformed.y - int(texcoordTransformed.y);
    if ((u > 0.5 && v > 0.5) || (u < 0.5 && v < 0.5)){
        return color0;
    } else {
        return color1;
    }
}
}