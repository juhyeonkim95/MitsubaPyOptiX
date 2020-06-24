#include <optixu/optixu_math_namespace.h>
#include "helpers.h"
#include "prd_struct.h"
using namespace optix;

rtBuffer<float, 2>              q_table;
rtBuffer<float, 2>              q_table_old;
rtBuffer<uint, 2>               visit_counts;

rtDeclareVariable(uint3,        unitCubeNumber, , );
rtDeclareVariable(uint2,        unitUVNumber, , );
rtDeclareVariable(float3,       unitCubeSize, , );

rtDeclareVariable(unsigned int,     q_table_update_method, , );
rtDeclareVariable(unsigned int,     q_value_sample_method, , );
rtDeclareVariable(float,     q_value_sample_constant, , );

static __host__ __device__ uint positionToIndex(float3 position)
{
    float3 abs_pos = make_float3(abs(position.x), abs(position.y), abs(position.z));
    uint3 idx = make_uint3(abs_pos / unitCubeSize);
    idx = clamp(idx, make_uint3(0), unitCubeNumber - make_uint3(1));
    uint idx_int = (idx.z * unitCubeNumber.y * unitCubeNumber.z) + (idx.y * unitCubeNumber.z) + idx.x;
    return idx_int;
}

static __host__ __device__ uint directionToIndex(float3 direction)
{
    float2 uv = mapDirectionToUV(direction);

    uint u = uint(uv.x * float(unitUVNumber.x));
    uint v = uint(uv.y * float(unitUVNumber.y));

    u = clamp(u, uint(0), unitUVNumber.x - 1);
    v = clamp(v, uint(0), unitUVNumber.y - 1);

    if(direction.y < 0){
        u += unitUVNumber.x;
    }
    uint idx = u * unitUVNumber.y + v;

    return idx;
}

static __host__ __device__ float3 getDirectionFrom(uint index, float2 offset)
{
    unsigned int u_index = (index / unitUVNumber.y);
    unsigned int v_index = (index % unitUVNumber.y);
    bool inverted = false;
    if (u_index > unitUVNumber.x){
        u_index -= unitUVNumber.x;
        inverted = true;
    }
    float u_index_r = (float(u_index) + offset.x)/(float(unitUVNumber.x));
    float v_index_r = (float(v_index) + offset.y)/(float(unitUVNumber.y));
    float3 random_direction = mapUVToDirection(make_float2(u_index_r, v_index_r));
    if (inverted){
        random_direction.y *= -1;
    }
    return random_direction;
}


static __host__ __device__ float getQValue(float3 position, float3 direction)
{
    uint positionIndex = positionToIndex(position);
    uint rayIndex = directionToIndex(direction);
    return q_table[make_uint2(positionIndex, rayIndex)];
}

static __host__ __device__ void setQValue(float3 position, float3 direction, float value)
{
    uint positionIndex = positionToIndex(position);
    uint rayIndex = directionToIndex(direction);
    q_table[make_uint2(positionIndex, rayIndex)] = value;
}

static __host__ __device__ uint updateVisit(float3 position, float3 direction)
{
    uint positionIndex = positionToIndex(position);
    uint rayIndex = directionToIndex(direction);
    uint2 idx = make_uint2(positionIndex, rayIndex);
    float v = visit_counts[idx];
    visit_counts[idx] = v + 1;

    return v;
}

static __host__ __device__ float getPolicyQValue(unsigned int positionIndex, unsigned int i){
    float target_value = 0;
    float bias_p = 0.01f;
    if(q_value_sample_method == 1){
        target_value = q_table_old[make_uint2(positionIndex, i)] + bias_p;
    } else if (q_value_sample_method == 2){
        target_value = q_table_old[make_uint2(positionIndex, i)] + bias_p;
        target_value = pow(target_value, q_value_sample_constant);
        //target_value * target_value;
    }
    return target_value;
}

static __host__ __device__ Sample_info sampleScatteringDirectionProportionalToQ(float3 position, float3 normal, bool considerCosineTerm, unsigned int &seed)
{
    uint positionIndex = positionToIndex(position);
    uint UVN = unitUVNumber.x * unitUVNumber.y;
    Sample_info sample_info;
    float q_value_sum = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        //float3 v_i = unitUVVectors[i];
        float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
        float cos_theta = dot(normal, v_i);
        if(cos_theta > 0.0f){
            float c = considerCosineTerm?cos_theta:1.0;
            q_value_sum += getPolicyQValue(positionIndex, i) * c;
        }
    }

    float random_v = rnd(seed) * q_value_sum;
    float accumulated_value = 0;
    unsigned int index = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        //float3 v_i = unitUVVectors[i];
        float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
        float cos_theta = dot(normal, v_i);
        if(cos_theta > 0.0f){
            float c = considerCosineTerm?cos_theta:1.0;
            accumulated_value += getPolicyQValue(positionIndex, i) * c;
            if(accumulated_value >= random_v){
                index = i;
                break;
            }
        }
    }


    //rtPrintf("attenuation %f \n",q_value_sum);
    float3 random_direction;
    while(true){
        random_direction = getDirectionFrom(index, make_float2(rnd(seed), rnd(seed)));
        if(dot(random_direction, normal) > 0.0f){
            break;
        }
    }

    float3 v_i = getDirectionFrom(index, make_float2(0.5, 0.5));
    float cos_theta = dot(normal, v_i);
    float c = considerCosineTerm?cos_theta:1.0;

    float p_w = (getPolicyQValue(positionIndex, index) * c) / q_value_sum;

    sample_info.direction = random_direction;
    sample_info.p_w = p_w;
    return sample_info;
}

static __host__ __device__ Sample_info sampleScatteringDirectionMaximumQ(float3 position, float3 normal, unsigned int &seed)
{
    uint positionIndex = positionToIndex(position);
    uint UVN = unitUVNumber.x * unitUVNumber.y;
    Sample_info sample_info;

    float q_value_sum = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        //float3 v_i = unitUVVectors[i];
        float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
        if(dot(normal, v_i) > 0.0f){
            q_value_sum += q_table_old[make_uint2(positionIndex, i)];
        }
    }

    float q_value_max = 0;
    unsigned int index = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        //float3 v_i = unitUVVectors[i];
        float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
        if(dot(normal, v_i) > 0.0f){
            float v = q_table_old[make_uint2(positionIndex, i)];
            if(q_value_max < v){
                q_value_max = v;
                index = i;
            }
        }
    }

    //rtPrintf("attenuation %f \n",q_value_sum);
    float3 random_direction;
    while(true){
        random_direction = getDirectionFrom(index, make_float2(rnd(seed), rnd(seed)));
        if(dot(random_direction, normal) > 0.0f){
            break;
        }
    }

    float p_w = q_table_old[make_uint2(positionIndex, index)] / q_value_sum;

    sample_info.direction = random_direction;
    sample_info.p_w = p_w;
    return sample_info;
}

static __host__ __device__ float getExpectedQValue(float3 position, float3 normal)
{
    uint positionIndex = positionToIndex(position);
    uint UVN = unitUVNumber.x * unitUVNumber.y;

    float q_value_sum = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        //float3 v_i = unitUVVectors[i];
        float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
        q_value_sum += (q_table[make_uint2(positionIndex, i)] * max(0.0, dot(normal, v_i)));
    }
    q_value_sum = q_value_sum / float(UVN);
    return q_value_sum;
}

static __host__ __device__ float getMaximumQValue(float3 position, float3 normal)
{
    uint positionIndex = positionToIndex(position);
    uint UVN = unitUVNumber.x * unitUVNumber.y;

    float q_value_max = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        //float3 v_i = unitUVVectors[i];
        float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
        float v = (q_table[make_uint2(positionIndex, i)] * max(0.0, dot(normal, v_i)));
        if(v > q_value_max){
            q_value_max = v;
        }
    }
    return q_value_max;
}

static __host__ __device__ Sample_info sampleScatteringDirectionProportionalToQVolume(float3 position, unsigned int &seed)
{
    uint positionIndex = positionToIndex(position);
    uint UVN = unitUVNumber.x * unitUVNumber.y;
    Sample_info sample_info;
    float q_value_sum = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        //float3 v_i = unitUVVectors[i];
        float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
        q_value_sum += getPolicyQValue(positionIndex, i);
    }

    float random_v = rnd(seed) * q_value_sum;
    float accumulated_value = 0;
    unsigned int index = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        //float3 v_i = unitUVVectors[i];
        float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
        accumulated_value += getPolicyQValue(positionIndex, i);
        if(accumulated_value >= random_v){
            index = i;
            break;
        }
    }
    float3 random_direction;
    random_direction = getDirectionFrom(index, make_float2(rnd(seed), rnd(seed)));

    float3 v_i = getDirectionFrom(index, make_float2(0.5, 0.5));

    float p_w = (getPolicyQValue(positionIndex, index)) / q_value_sum;

    sample_info.direction = random_direction;
    sample_info.p_w = p_w;
    return sample_info;
}

static __host__ __device__ Sample_info sampleScatteringDirectionProportionalToQVolumeHG(float3 position, float3 direction, float g, unsigned int &seed)
{
    uint positionIndex = positionToIndex(position);
    uint UVN = unitUVNumber.x * unitUVNumber.y;
    Sample_info sample_info;
    float q_value_sum = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        //float3 v_i = unitUVVectors[i];
        float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
        float hg = HG_phase_function(g, dot(direction, v_i));
        q_value_sum += getPolicyQValue(positionIndex, i) * hg;
    }

    float random_v = rnd(seed) * q_value_sum;
    float accumulated_value = 0;
    unsigned int index = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        //float3 v_i = unitUVVectors[i];
        float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
        float hg = HG_phase_function(g, dot(direction, v_i));
        accumulated_value += getPolicyQValue(positionIndex, i) * hg;
        if(accumulated_value >= random_v){
            index = i;
            break;
        }
    }
    float3 random_direction;
    random_direction = getDirectionFrom(index, make_float2(rnd(seed), rnd(seed)));

    float3 v_i = getDirectionFrom(index, make_float2(0.5, 0.5));
    float hg = HG_phase_function(g, dot(direction, v_i));
    float p_w = (getPolicyQValue(positionIndex, index) * hg) / q_value_sum;

    sample_info.direction = random_direction;
    sample_info.p_w = p_w;
    return sample_info;
}

static __host__ __device__ float getNextQValue(float3 position, float3 normal, float3 direction)
{
    float new_value;
    // Expected SARSA
    if(q_table_update_method == 0){
        new_value = 2 * getExpectedQValue(position, normal);
    }
    // Q-learning
    else if(q_table_update_method == 1){
        new_value = getMaximumQValue(position, normal);
    }
    // SARSA
    else if(q_table_update_method == 2){
        new_value = getQValue(position, direction);
    }
    return new_value;
}