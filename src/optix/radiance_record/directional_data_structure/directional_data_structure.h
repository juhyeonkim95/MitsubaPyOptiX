#pragma once
#ifndef DIRECTIONAL_DATA_STRUCTURE_H
#define DIRECTIONAL_DATA_STRUCTURE_H

#include "optix/radiance_record/directional_data_structure/quad_tree.h"
#include "optix/radiance_record/directional_data_structure/grid.h"
#include "optix/radiance_record/directional_mapping/directional_mapping.h"

#include "optix/common/rt_function.h"

using namespace optix;

// Non path-guided sampling
#define DIRECTIONAL_STRUCTURE_GRID 0
#define DIRECTIONAL_STRUCTURE_QUADTREE 1

rtDeclareVariable(unsigned int,     directional_data_structure_type, , );

RT_FUNCTION uint UVToIndex(const uint pos_index, const float2 &uv)
{
    switch(directional_data_structure_type){
    case DIRECTIONAL_STRUCTURE_GRID: return grid::UVToIndex(uv);
    case DIRECTIONAL_STRUCTURE_QUADTREE: return quad_tree::UVToIndex(pos_index, uv);
    }
    return 0;
}

RT_FUNCTION uint DirectionToIndex(const uint pos_index, const float3 &direction)
{
    float2 uv = DirectionToUV(direction);
    return UVToIndex(pos_index, uv);
}

#endif
