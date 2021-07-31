#include <optixu/optixu_math_namespace.h>
#include "optix/radiance_record/q_table.h"
#include "optix/radiance_record/directional_mapping/cylindrical_mapping.h"

using namespace optix;

rtBuffer<unsigned int, 2>     dtree_visit_count;
rtBuffer<unsigned int, 2>     dtree_index_array;
rtBuffer<unsigned int, 2>     dtree_rank_array;

namespace quad_tree{
RT_FUNCTION uint UVToIndex(const uint pos_index, const float2 &p_)
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

RT_FUNCTION float2 IndexToUV(const uint pos_index, const uint dir_index, const float2& offset)
{
    // not required!!
    return make_float2(1.0);
}

RT_FUNCTION Sample_info Sample(const uint pos_index, unsigned int &seed)
{
    float pdf = 1;
    float bias = 0.0;
    uint2 pos_dir_idx = make_uint2(0, pos_index);
    float2 p = make_float2(0);
    float size = 1;

    while(true){
        float sum = (q_table[pos_dir_idx] + bias) * 4;
        // sum = 0.0;
        // Leaf node
        if(sum <= 0.0 || dtree_index_array[pos_dir_idx] == 0){
            p += make_float2(rnd(seed), rnd(seed)) * size;
            break;
        }

        float accumulated_value = 0;
        float random_value = rnd(seed) * sum;

        for(int i=0; i<4; i++){
            uint child_idx = 4 * dtree_rank_array[pos_dir_idx] + i + 1;
            float q_value = q_table[make_uint2(child_idx, pos_index)] + bias;
            accumulated_value += q_value;
            if(random_value <= accumulated_value){
                p.x += (i % 2 == 1) ? size * 0.5 : 0;
                p.y += (i >= 2) ? size * 0.5 : 0;
                pos_dir_idx.x = child_idx;
                pdf *= (4 * q_value / sum);
                break;
            }
        }
        size *= 0.5f;
    }

    pdf *= 1 / (4 * M_PIf);
    Sample_info sample_info;
    sample_info.pdf = pdf;
    sample_info.direction = cylindrical_mapping::UVToDirection(p);
    return sample_info;
}

RT_FUNCTION float Pdf(const uint pos_index, const float3 &direction)
{
    float2 p = cylindrical_mapping::DirectionToUV(direction);

    uint child = 0;
    float pdf = 1;
    uint2 pos_dir_idx = make_uint2(0, pos_index);

    pdf = 1;

    while(true){
        float sum = q_table[pos_dir_idx] * 4;

        if(sum <= 0.0 || dtree_index_array[pos_dir_idx] == 0){
            break;
        }

        // Go to child node
        bool x = p.x > 0.5;
        bool y = p.y > 0.5;
        uint child_idx = (y << 1) | x;
        p.x = x ? 2 * p.x - 1 : 2 * p.x;
        p.y = y ? 2 * p.y - 1 : 2 * p.y;
        child = 4 * dtree_rank_array[pos_dir_idx] + child_idx + 1;
        float q_value = q_table[make_uint2(child, pos_index)];
        pdf *= (4 * q_value / sum);

        pos_dir_idx.x = child;
    }
    pdf *= 1 / (4 * M_PIf);
    return pdf;
}
}
