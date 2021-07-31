#pragma once
#ifndef CYLINDRICAL_MAPPING_H
#define CYLINDRICAL_MAPPING_H

#include <optixu/optixu_math_namespace.h>
#include "optix/common/rt_function.h"

namespace cylindrical_mapping
{
    RT_FUNCTION float3 UVToDirection(const float2 &uv){
        const float cosTheta = 2 * uv.x - 1;
        const float phi = 2 * M_PI * uv.y;
        const float sinTheta = sqrt(1 - cosTheta * cosTheta);
        float sinPhi = sinf(phi);
        float cosPhi = cosf(phi);
        return make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
    }

    RT_FUNCTION float2 DirectionToUV(const float3 &direction){
        const float cosTheta = min(max(direction.z, -1.0f), 1.0f);
        float phi = atan2(direction.y, direction.x);
        phi = phi < 0 ? phi + 2.0 * M_PIf : phi;
        return make_float2((cosTheta + 1) * 0.5, phi / (2 * M_PIf));
    }
}

#endif
