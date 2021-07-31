#pragma once
#ifndef RADIANCE_RECORD_H
#define RADIANCE_RECORD_H
#include "optix/radiance_record/spatial_data_structure/spatial_data_structure.h"
#include "optix/radiance_record/directional_data_structure/directional_data_structure.h"

#include "optix/common/rt_function.h"
using namespace optix;


namespace radiance_record
{
RT_FUNCTION float Pdf(const float3 &position, const float3 &normal,
const float3 &direction)
{
    const uint pos_index = positionToIndex(position);

    switch(directional_data_structure_type){
    case DIRECTIONAL_STRUCTURE_QUADTREE: return quad_tree::Pdf(pos_index, direction);
    case DIRECTIONAL_STRUCTURE_GRID: return grid::Pdf(pos_index, direction);
    }
    return 0.0f;
}

RT_FUNCTION Sample_info Sample(const float3 &position, const float3 &normal,
    unsigned int &seed)
{
    const uint pos_index = positionToIndex(position);

    switch(directional_data_structure_type){
    case DIRECTIONAL_STRUCTURE_QUADTREE: return quad_tree::Sample(pos_index, seed);
    case DIRECTIONAL_STRUCTURE_GRID: return grid::Sample(pos_index, seed);
    }
    Sample_info sample_info;
    return sample_info;
}
}

RT_FUNCTION void accumulateQValue(const float3 &position, const float3 &direction, float value)
{
    uint pos_index = positionToIndex(position);
    uint dir_index = DirectionToIndex(pos_index, direction);
    uint2 idx = make_uint2(dir_index, pos_index);
    atomicAdd(&q_table_accumulated[idx], value);
    atomicAdd(&q_table_visit_counts[idx], 1);
}

#endif
