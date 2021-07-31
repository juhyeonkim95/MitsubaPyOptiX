#include <optixu/optixu_math_namespace.h>
#include "optix/common/rt_function.h"

rtBuffer<unsigned int>     stree_visit_count;
rtBuffer<unsigned int>     stree_child_array;
rtBuffer<unsigned int>     stree_parent_array;
rtBuffer<unsigned int>     stree_axis_array;
rtBuffer<unsigned int>     stree_leaf_index_array;

namespace binary_tree
{
    RT_FUNCTION uint normalizedPositionToIndex(const float3 &p_)
    {
        float3 p = p_;
        uint idx = 0;
        uint child_local_idx = 0;
        while(true){
            // Leaf node
            if(stree_child_array[idx] == 0){
                break;
            }

            uint axis = stree_axis_array[stree_child_array[idx]];
            switch(axis){
            case 0: if(p.x < 0.5f){p.x = 2 * p.x; child_local_idx = 0;} else{p.x = 2 * p.x - 1; child_local_idx = 1;} break;
            case 1: if(p.y < 0.5f){p.y = 2 * p.y; child_local_idx = 0;} else{p.y = 2 * p.y - 1; child_local_idx = 1;} break;
            case 2: if(p.z < 0.5f){p.z = 2 * p.z; child_local_idx = 0;} else{p.z = 2 * p.z - 1; child_local_idx = 1;} break;
            }

            // Go to child node
            idx = stree_child_array[idx] + child_local_idx;
        }

        // to leaf node index
        // return idx;
        return stree_leaf_index_array[idx];
    }
}
