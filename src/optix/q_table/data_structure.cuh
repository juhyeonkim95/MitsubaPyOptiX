#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include "optix/common/rt_function.h"

/// Octree related

rtBuffer<unsigned int>     stree_visit_count;
rtBuffer<unsigned int>     stree_index_array;
rtBuffer<unsigned int>     stree_rank_array;
//rtBuffer<unsigned int>     stree_rank_leaf_array;



using namespace optix;
RT_FUNCTION void incrementPositionInSTree(float3 p_)
{
    // Leaf node
    //if(stree_index_array[idx] == 0){
    //    atomicAdd(&stree_visit_count[idx], 1);
    //}
}

RT_FUNCTION uint positionToIndexOctTree(const float3 &p_)
{
    float3 p = p_;
    uint idx = 0;
    uint child = 0;
    while(true){
        // Go to child node
        bool x = p.x > 0.5;
        bool y = p.y > 0.5;
        bool z = p.z > 0.5;
        uint child_idx = (z << 2) | (y << 1) | x;
        p.x = x ? 2 * p.x - 1 : 2 * p.x;
        p.y = y ? 2 * p.y - 1 : 2 * p.y;
        p.z = z ? 2 * p.z - 1 : 2 * p.z;
        child = 8 * stree_rank_array[idx] + child_idx + 1;

        // No child node
        if(stree_index_array[child] == 0)
            break;
        idx = child;
    }
    return stree_rank_array[idx];
}



/// Grid related

rtDeclareVariable(float3,       scene_bbox_min, , );
rtDeclareVariable(float3,       scene_bbox_max, , );
rtDeclareVariable(float3,       scene_bbox_extent, , );
rtDeclareVariable(uint3,        unitCubeNumber, , );

RT_FUNCTION uint positionToIndexVoxel(const float3 &p)
{
    uint3 idx = make_uint3(p.x * unitCubeNumber.x, p.y * unitCubeNumber.y, p.z * unitCubeNumber.z);
    idx = clamp(idx, make_uint3(0), unitCubeNumber - make_uint3(1));
    uint idx_int = (idx.x * unitCubeNumber.y * unitCubeNumber.z) + (idx.y * unitCubeNumber.z) + idx.z;
    return idx_int;
}

rtDeclareVariable(unsigned int,     spatial_table_type, , );

RT_FUNCTION uint positionToIndex(const float3 &position){
    float3 normalized_position = (position - scene_bbox_min) / scene_bbox_extent;
    normalized_position = clamp(normalized_position, 0, 1);

    switch(spatial_table_type){
    case 0: return positionToIndexVoxel(normalized_position);
    case 1: return positionToIndexOctTree(normalized_position);
    default: return 0;
    }
}


// *********************
// Direction
// *********************

rtDeclareVariable(uint2,        unitUVNumber, , );

RT_FUNCTION uint directionToIndexGrid(const float3 &direction)
{
    float2 uv = mapDirectionToUV(direction);

    uint u = clamp(uint(uv.x * float(unitUVNumber.x)), uint(0), unitUVNumber.x - 1);
    uint v = clamp(uint(uv.y * float(unitUVNumber.y)), uint(0), unitUVNumber.y - 1);

    if(direction.y < 0){
        u += unitUVNumber.x;
    }

    uint idx_int = (u * unitUVNumber.y) + v;
    return idx_int;
}

RT_FUNCTION uint directionToIndexGrid2(const float2 &uv)
{
    uint u = clamp(uint(uv.x * float(unitUVNumber.x)), uint(0), unitUVNumber.x - 1);
    uint v = clamp(uint(uv.y * float(unitUVNumber.y)), uint(0), unitUVNumber.y - 1);
    uint idx_int = (u * unitUVNumber.y) + v;
    return idx_int;
}

rtBuffer<unsigned int, 2>     dtree_visit_count;
rtBuffer<unsigned int, 2>     dtree_index_array;
rtBuffer<unsigned int, 2>     dtree_rank_array;

RT_FUNCTION uint directionToIndexQuadTree(const uint pos_index, const float2 &p_)
{
    float2 p = p_;
    uint child = 0;
    uint2 pos_dir_idx = make_uint2(0, pos_index);

    while(true){
        // No child node
        if(dtree_index_array[pos_dir_idx] == 0)
            break;

        // Go to child node
        bool x = p.x > 0.5;
        bool y = p.y > 0.5;
        uint child_idx = (y << 1) | x;
        p.x = x ? 2 * p.x - 1 : 2 * p.x;
        p.y = y ? 2 * p.y - 1 : 2 * p.y;
        child = 4 * dtree_rank_array[pos_dir_idx] + child_idx + 1;

        pos_dir_idx.x = child;
    }
    return pos_dir_idx.x;
}


rtDeclareVariable(unsigned int,     directional_table_type, , );
rtDeclareVariable(unsigned int,     directional_mapping_method, , );

RT_FUNCTION uint directionToIndex(const uint pos_index, const float3 &direction){
    float2 uv = mapDirectionToCanonical(direction);
    switch(directional_table_type){
    case 0:
        switch(directional_mapping_method){
        case 0: return directionToIndexGrid(direction);
        case 1: return directionToIndexGrid2(uv);
        }
    case 1:
        return directionToIndexQuadTree(pos_index, uv);
    default: return 0;
    }
}
