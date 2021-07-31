#pragma once
#ifndef SPATIAL_DATA_STRUCTURE_H
#define SPATIAL_DATA_STRUCTURE_H

#include "optix/radiance_record/spatial_data_structure/binary_tree.h"
#include "optix/radiance_record/spatial_data_structure/voxel.h"

#include "optix/common/rt_function.h"

using namespace optix;

// Non path-guided sampling
#define SPATIAL_STRUCTURE_VOXEL 0
#define SPATIAL_STRUCTURE_OCTREE 1
#define SPATIAL_STRUCTURE_BINARY_TREE 2

// For normalization
rtDeclareVariable(float3,       scene_bbox_min, , );
rtDeclareVariable(float3,       scene_bbox_max, , );
rtDeclareVariable(float3,       scene_bbox_extent, , );

rtDeclareVariable(unsigned int,     spatial_data_structure_type, , );

RT_FUNCTION uint positionToIndex(const float3 &position)
{
    float3 normalized_position = (position - scene_bbox_min) / scene_bbox_extent;
    normalized_position = clamp(normalized_position, 0, 1);

    switch(spatial_data_structure_type){
    case SPATIAL_STRUCTURE_BINARY_TREE: return binary_tree::normalizedPositionToIndex(normalized_position);
    case SPATIAL_STRUCTURE_VOXEL: return voxel::normalizedPositionToIndex(normalized_position);
    default: return 0;
    }
}
#endif
