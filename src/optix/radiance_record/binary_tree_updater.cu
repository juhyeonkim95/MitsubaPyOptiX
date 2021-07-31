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

rtDeclareVariable(uint,         launch_index, rtLaunchIndex, );


rtBuffer<unsigned int>     stree_visit_count;
rtBuffer<unsigned int>     stree_child_array;
rtBuffer<unsigned int>     stree_parent_array;
rtBuffer<unsigned int>     stree_axis_array;
rtBuffer<unsigned int>     stree_size;

rtDeclareVariable(int,  stree_split_threshold, , );

RT_FUNCTION void copy_child(int o, int d){
    int current_size = dtree_current_size_array[o];
    dtree_current_size_array[d] = current_size;
    for(int i=0; i<current_size; i++){
        dtree_index_array[make_uint2(i, d)] = dtree_index_array[make_uint2(i, o)];
        dtree_rank_array[make_uint2(i, d)] = dtree_rank_array[make_uint2(i, o)];
        dtree_depth_array[make_uint2(i, d)] = dtree_depth_array[make_uint2(i, o)];
        dtree_select_array[make_uint2(i, d)] = dtree_select_array[make_uint2(i, o)];
        q_table[make_uint2(i, d)] = q_table[make_uint2(i, o)];
    }
}

RT_FUNCTION void subdivide(int parent, int size){
    stree_child_array[parent] = size;
    for(int i=0; i<2; i++){
        int child = size + i;
        stree_axis_array[child] = (stree_axis_array[parent] + 1) % 3;
        stree_parent_array[child] = parent;
        stree_visit_count[child] = stree_visit_count[parent] / 2;
        // copy_child(parent, child);
    }
    //size += 2;
}

RT_PROGRAM void spatial_binary_tree_updater(){

    int max_size = 512*16;
    int current_size = stree_size[launch_index];

    for(int i=0; i<stree_size[0]; i++){
        if(current_size >= max_size){
            break;
        }
        if(stree_child_array[i] == 0){
            int visited_count = stree_visit_count[i];

            // subdivide(i, current_size);
            current_size += 2;

            if(visited_count > 100){
                //subdivide(i, current_size-2);
                //current_size += 2;
            }
        }
    }
    stree_size[0] = current_size;
    //rtVariableSet1i(stree_size, current_size);
}
