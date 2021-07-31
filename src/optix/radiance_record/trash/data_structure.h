//#pragma once
//#ifndef DATA_STRUCTURE_H
//#define DATA_STRUCTURE_H
//
//#include <optixu/optixu_math_namespace.h>
//#include <optixu/optixu_matrix_namespace.h>
//#include "optix/common/rt_function.h"
//
//
//using namespace optix;
//RT_FUNCTION void incrementPositionInSTree(const float3 &p)
//{
//    uint positionIndex = positionToIndex(p);
//    atomicAdd(&stree_visit_count[positionIndex], 1);
//}
//
//
//// Directional data structure
//
//// *********************
//// (1) Grid
//// *********************
//
//rtDeclareVariable(uint2,        unitUVNumber, , );
//
//RT_FUNCTION uint directionToIndexGrid(const float3 &direction)
//{
//    float2 uv = mapDirectionToUV(direction);
//
//    uint u = clamp(uint(uv.x * float(unitUVNumber.x)), uint(0), unitUVNumber.x - 1);
//    uint v = clamp(uint(uv.y * float(unitUVNumber.y)), uint(0), unitUVNumber.y - 1);
//
//    if(direction.y < 0){
//        u += unitUVNumber.x;
//    }
//
//    uint idx_int = (u * unitUVNumber.y) + v;
//    return idx_int;
//}
//
//RT_FUNCTION uint directionToIndexGrid2(const float2 &uv)
//{
//    uint u = clamp(uint(uv.x * float(unitUVNumber.x)), uint(0), unitUVNumber.x - 1);
//    uint v = clamp(uint(uv.y * float(unitUVNumber.y)), uint(0), unitUVNumber.y - 1);
//    uint idx_int = (u * unitUVNumber.y) + v;
//    return idx_int;
//}
//
//rtBuffer<unsigned int, 2>     dtree_visit_count;
//rtBuffer<unsigned int, 2>     dtree_index_array;
//rtBuffer<unsigned int, 2>     dtree_rank_array;
//
//// *********************
//// (2) QuadTree
//// *********************
//
//RT_FUNCTION uint directionToIndexQuadTree(const uint pos_index, const float2 &p_)
//{
//    float2 p = p_;
//    uint child = 0;
//    uint2 pos_dir_idx = make_uint2(0, pos_index);
//
//    while(true){
//        // No child node
//        if(dtree_index_array[pos_dir_idx] == 0)
//            break;
//
//        // Go to child node
//        bool x = p.x > 0.5;
//        bool y = p.y > 0.5;
//        uint child_idx = (y << 1) | x;
//        p.x = x ? 2 * p.x - 1 : 2 * p.x;
//        p.y = y ? 2 * p.y - 1 : 2 * p.y;
//        child = 4 * dtree_rank_array[pos_dir_idx] + child_idx + 1;
//
//        pos_dir_idx.x = child;
//    }
//    return pos_dir_idx.x;
//}
//
//
//rtDeclareVariable(unsigned int,     directional_table_type, , );
//rtDeclareVariable(unsigned int,     directional_mapping_method, , );
//
//RT_FUNCTION uint directionToIndex(const uint pos_index, const float3 &direction){
//    float2 uv = mapDirectionToCanonical(direction);
//    switch(directional_table_type){
//    case 0:
//        switch(directional_mapping_method){
//        case 0: return directionToIndexGrid(direction);
//        case 1: return directionToIndexGrid2(uv);
//        }
//    case 1:
//        return directionToIndexQuadTree(pos_index, uv);
//    default: return 0;
//    }
//}
//
//RT_FUNCTION float3 indexToDirection(const uint index, const float2 &offset)
//{
//    unsigned int u_index = (index / unitUVNumber.y);
//    unsigned int v_index = (index % unitUVNumber.y);
//    bool inverted = false;
//    if (u_index > unitUVNumber.x){
//        u_index -= unitUVNumber.x;
//        inverted = true;
//    }
//    float u_index_r = (float(u_index) + offset.x)/(float(unitUVNumber.x));
//    float v_index_r = (float(v_index) + offset.y)/(float(unitUVNumber.y));
//    float3 random_direction = mapUVToDirection(make_float2(u_index_r, v_index_r));
//    if (inverted){
//        random_direction.y *= -1;
//    }
//    return random_direction;
//}
//
//#endif