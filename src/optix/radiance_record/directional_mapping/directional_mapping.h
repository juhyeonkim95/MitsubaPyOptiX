#pragma once
#ifndef DIRECTION_UV_MAPPING_H
#define DIRECTION_UV_MAPPING_H

#include <optixu/optixu_math_namespace.h>
#include "optix/common/rt_function.h"
#include "optix/radiance_record/directional_mapping/shirley_mapping.h"
#include "optix/radiance_record/directional_mapping/cylindrical_mapping.h"
#include "optix/app_config.h"

rtDeclareVariable(unsigned int,     directional_mapping_method, , );


//#define DIRECTION_UV_MAPPING_SHIRLEY 0
//#define DIRECTION_UV_MAPPING_CYLINDRICAL 1

// this maps [0,1]^2 to sphere
RT_FUNCTION float3 UVToDirection(const float2 &uv)
{
#if DIRECTION_UV_MAPPING_TYPE == DIRECTION_UV_MAPPING_SHIRLEY
    return shirley_mapping::UVToDirection(uv);
#elif DIRECTION_UV_MAPPING_TYPE == DIRECTION_UV_MAPPING_CYLINDRICAL
    return cylindrical_mapping::UVToDirection(uv);
#else
    return make_float3(0);
#endif
}

// this maps sphere to [0,1]^2 to uv
RT_FUNCTION float2 DirectionToUV(const float3 &direction)
{
#if DIRECTION_UV_MAPPING_TYPE == DIRECTION_UV_MAPPING_SHIRLEY
    return shirley_mapping::DirectionToUV(direction);
#elif DIRECTION_UV_MAPPING_TYPE == DIRECTION_UV_MAPPING_CYLINDRICAL
    return cylindrical_mapping::DirectionToUV(direction);
#else
    return make_float2(0);
#endif
}

#endif
