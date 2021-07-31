/*
 * Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <optixu/optixu_math_namespace.h>
#include <optix_host.h>

//#include "optix/common/helpers.h"
//#include "optix/q_table/data_structure.cuh"
#include "optix/common/rt_function.h"


rtBuffer<unsigned int, 2>     dtree_index_array;
rtBuffer<unsigned int, 2>     dtree_rank_array;
rtBuffer<unsigned int, 2>     dtree_depth_array;
rtBuffer<unsigned int, 2>     dtree_select_array;
rtBuffer<unsigned int, 1>     dtree_current_size_array;

// rtBuffer<rtBufferId<float, 1>, 1> dtree_value_array;

rtBuffer<float, 2>     q_table;

using namespace optix;

//struct QuadTree {
//    rtBufferId<unsigned int, 1>     index_array;
//    rtBufferId<unsigned int, 1>     rank_array;
//    rtBufferId<unsigned int, 1>     depth_array;
//    rtBufferId<unsigned int, 1>     select_array;
//    rtBufferId<float, 1>            value_array;
//    uint current_size;
//};
//
//rtBuffer<QuadTree> quad_trees;

rtDeclareVariable(float,     quad_tree_update_threshold, , );
rtDeclareVariable(uint,         launch_index, rtLaunchIndex, );

RT_FUNCTION float get_irradiance(){
    float irradiance = 0;
    uint current_size = dtree_current_size_array[launch_index];
    for(int i=0; i<current_size; i++){
        if(dtree_index_array[make_uint2(i, launch_index)] == 0){
            float value = q_table[make_uint2(i, launch_index)];
            unsigned int depth = dtree_depth_array[make_uint2(i, launch_index)];
            irradiance += value * pow(0.25, depth);
        }
    }
    return irradiance;
}

RT_FUNCTION uint child(uint idx, uint child_idx){
    return 4 * dtree_rank_array[make_uint2(idx, launch_index)] + child_idx + 1;
}

RT_FUNCTION void build_rank_array(){
    uint sum = 0;
    uint current_size = dtree_current_size_array[launch_index];
    for(int i=0; i< current_size; i++){
        dtree_rank_array[make_uint2(i, launch_index)] = sum;
        sum += dtree_index_array[make_uint2(i, launch_index)];
    }

    for(int i=0; i< current_size; i++){
        if(dtree_index_array[make_uint2(i, launch_index)] == 1){
            for(int j=0; j<4; j++){
                uint child_j = child(i, j);
                dtree_select_array[make_uint2(child_j, launch_index)] = i;
            }
        }
    }
}

RT_FUNCTION uint parent(uint child_idx){
    return dtree_select_array[make_uint2(child_idx, launch_index)];
}

RT_FUNCTION void update_parent_radiance(){
    uint current_size = dtree_current_size_array[launch_index];
    for(int i=current_size - 1; i>=1 ; i--){
        uint parent_id = parent(i);
        q_table[make_uint2(parent_id, launch_index)] += q_table[make_uint2(i, launch_index)] * 0.25;
    }
}

RT_PROGRAM void quad_tree_updater(){
    float total_irradiance = get_irradiance();
    uint current_size = dtree_current_size_array[launch_index];
    uint current_size_original = current_size;

    const uint MAX_DEQUEUE_SIZE = 1024;
    const uint MAX_QUADTREE_SIZE = 512;

    uint index_dequeue[MAX_DEQUEUE_SIZE];
    float value_dequeue[MAX_DEQUEUE_SIZE];
    uint depth_dequeue[MAX_DEQUEUE_SIZE];

    uint index_array[MAX_QUADTREE_SIZE];
    float value_array[MAX_QUADTREE_SIZE];
    uint depth_array[MAX_QUADTREE_SIZE];

    uint current_q_size = 0;
    uint current_q_pointer_left = 0;
    uint current_q_pointer_right = 0;
    uint current_array_size = 0;

    index_dequeue[0] = 0;
    value_dequeue[0] = q_table[make_uint2(0, launch_index)];
    depth_dequeue[0] = 0;
    current_q_pointer_right += 1;
    current_q_size += 1;

    while(current_q_size > 0){
        uint node = index_dequeue[current_q_pointer_left];
        float val = value_dequeue[current_q_pointer_left];
        uint depth = depth_dequeue[current_q_pointer_left];

        current_q_pointer_left += 1;
        current_q_size -= 1;

        depth_array[current_array_size] = depth;

        if(node < current_size_original){
            float value = q_table[make_uint2(node, launch_index)];

            // internal node
            if(dtree_index_array[make_uint2(node ,launch_index)] == 1){
                index_array[current_array_size] = 1;
                value_array[current_array_size] = 0.0f;
                current_array_size += 1;

                for(uint i=0; i<4; i++){
                    uint child_id = child(node, i);
                    index_dequeue[current_q_pointer_right] = child_id;
                    value_dequeue[current_q_pointer_right] = value / 4;
                    depth_dequeue[current_q_pointer_right] = depth + 1;
                    current_q_pointer_right += 1;
                    current_q_size += 1;
                }
            }
            // leaf node
            else{
                float local_irradiance = value * pow(0.25, depth);
                // (local_irradiance > total_irradiance * threshold) &&

                if((local_irradiance > total_irradiance * quad_tree_update_threshold) && (current_size + 4 <= MAX_QUADTREE_SIZE)){
                    index_array[current_array_size] = 1;
                    value_array[current_array_size] = 0.0f;
                    current_array_size += 1;
                    for(uint i=0; i<4; i++){
                        index_dequeue[current_q_pointer_right] = current_size + i;
                        value_dequeue[current_q_pointer_right] = value / 4;
                        depth_dequeue[current_q_pointer_right] = depth + 1;
                        current_q_pointer_right += 1;
                        current_q_size += 1;
                    }
                    current_size += 4;

                } else {
                    index_array[current_array_size] = 0;
                    value_array[current_array_size] = value;
                    current_array_size += 1;
                }
            }
        }
        else {
            index_array[current_array_size] = 0;
            value_array[current_array_size] = val;
            current_array_size += 1;
        }
    }

    for(int i=0; i<current_size; i++){
        dtree_index_array[make_uint2(i, launch_index)] = index_array[i];
        q_table[make_uint2(i, launch_index)] = value_array[i];
        dtree_depth_array[make_uint2(i, launch_index)] = depth_array[i];
    }
    dtree_current_size_array[launch_index] = current_size;
    build_rank_array();
    update_parent_radiance();
}
