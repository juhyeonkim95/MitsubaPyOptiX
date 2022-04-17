#pragma once
#ifndef RADIANCE_RECORD_H
#define RADIANCE_RECORD_H
#include "optix/radiance_record/spatial_data_structure/spatial_data_structure.h"
#include "optix/radiance_record/directional_data_structure/directional_data_structure.h"

#include "optix/common/rt_function.h"
#include "optix/app_config.h"
using namespace optix;


namespace radiance_record
{
RT_FUNCTION float Pdf(const float3 &position, const float3 &normal,
const float3 &direction)
{
    const uint pos_index = positionToIndex(position);

#if DIRECTIONAL_DATA_STRUCTURE_TYPE == DIRECTIONAL_DATA_STRUCTURE_GRID
    return grid::Pdf(pos_index, direction);
#elif DIRECTIONAL_DATA_STRUCTURE_TYPE == DIRECTIONAL_DATA_STRUCTURE_QUADTREE
    return quad_tree::Pdf(pos_index, direction);
#else
    return 0;
#endif
}

RT_FUNCTION Sample_info Sample(const float3 &position, const float3 &normal,
    unsigned int &seed)
{
    const uint pos_index = positionToIndex(position);

#if DIRECTIONAL_DATA_STRUCTURE_TYPE == DIRECTIONAL_DATA_STRUCTURE_GRID
    return grid::Sample(pos_index, seed);
#elif DIRECTIONAL_DATA_STRUCTURE_TYPE == DIRECTIONAL_DATA_STRUCTURE_QUADTREE
    return quad_tree::Sample(pos_index, seed);
#else
    Sample_info sample_info;
    return sample_info;
#endif
}

RT_FUNCTION Sample_info SampleNormalSensitive(const float3 &position, const float3 &normal,
    unsigned int &seed)
{
    const uint pos_index = positionToIndex(position);
    return grid::SampleNormalSensitive(pos_index, normal, seed);
}
RT_FUNCTION Sample_info SampleNormalSensitiveReject(const float3 &position, const float3 &normal, const optix::Onb &onb,
    unsigned int &seed)
{
    const uint pos_index = positionToIndex(position);
    return grid::SampleNormalSensitiveReject(pos_index, normal, onb, seed);
}
}

RT_FUNCTION float getQValueFromPosDir(const float3 &position, const float3 &direction)
{
    uint pos_index = positionToIndex(position);
    uint dir_index = DirectionToIndex(pos_index, direction);
    uint2 idx = make_uint2(dir_index, pos_index);
    return q_table[idx];
}

RT_FUNCTION void accumulateQValue(const float3 &position, const float3 &direction, float value)
{
    uint pos_index = positionToIndex(position);
    uint dir_index = DirectionToIndex(pos_index, direction);
    uint2 idx = make_uint2(dir_index, pos_index);
    atomicAdd(&q_table_accumulated[idx], value);
    atomicAdd(&q_table_visit_counts[idx], 1);
}

RT_FUNCTION void accumulateQValueIndexed(const uint pos_index, const uint dir_index, float value)
{
    uint2 idx = make_uint2(dir_index, pos_index);
    atomicAdd(&q_table_accumulated[idx], value);
    atomicAdd(&q_table_visit_counts[idx], 1);
}

#endif
