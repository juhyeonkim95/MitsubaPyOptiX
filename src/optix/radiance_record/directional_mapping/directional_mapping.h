#pragma once
#ifndef DIRECTION_UV_MAPPING_H
#define DIRECTION_UV_MAPPING_H

#include <optixu/optixu_math_namespace.h>
#include "optix/common/rt_function.h"
#include "optix/radiance_record/directional_mapping/shirely_mapping.h"
#include "optix/radiance_record/directional_mapping/cylindrical_mapping.h"

rtDeclareVariable(unsigned int,     directional_mapping_method, , );


#define DIRECTION_UV_MAPPING_SHIRLEY 0
#define DIRECTION_UV_MAPPING_CYLINDRICAL 1

// this maps [0,1]^2 to sphere
RT_FUNCTION float3 UVToDirection(const float2 &uv)
{
    switch(directional_mapping_method){
    case DIRECTION_UV_MAPPING_SHIRLEY: return shirely_mapping::UVToDirection(uv);
    case DIRECTION_UV_MAPPING_CYLINDRICAL: return cylindrical_mapping::UVToDirection(uv);
    }
    return cylindrical_mapping::UVToDirection(uv);
}

// this maps sphere to [0,1]^2 to uv
RT_FUNCTION float2 DirectionToUV(const float3 &direction)
{
    switch(directional_mapping_method){
    case DIRECTION_UV_MAPPING_SHIRLEY: return shirely_mapping::DirectionToUV(direction);
    case DIRECTION_UV_MAPPING_CYLINDRICAL: return cylindrical_mapping::DirectionToUV(direction);
    }
    return cylindrical_mapping::DirectionToUV(direction);
}

#endif
