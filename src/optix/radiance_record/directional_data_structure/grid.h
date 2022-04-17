#include <optixu/optixu_math_namespace.h>
#include "optix/radiance_record/q_table.h"
#include "optix/radiance_record/directional_mapping/directional_mapping.h"
#include "optix/app_config.h"

using namespace optix;
rtDeclareVariable(uint2,        unitUVNumber, , );
rtDeclareVariable(float2,        unitUVNumber_inv, , );
rtBuffer<float3, 1>             unitUVVectors;

namespace grid{
RT_FUNCTION uint UVToIndex(const float2 &uv)
{
    uint u = clamp(uint(uv.x * float(unitUVNumber.x)), uint(0), unitUVNumber.x - 1);
    uint v = clamp(uint(uv.y * float(unitUVNumber.y)), uint(0), unitUVNumber.y - 1);
    uint idx_int = (u * unitUVNumber.y) + v;
    return idx_int;
}

RT_FUNCTION float2 IndexToUV(const uint dir_index, const float2& offset)
{
    unsigned int u_index = (dir_index / unitUVNumber.y);
    unsigned int v_index = (dir_index % unitUVNumber.y);

    return (make_float2(u_index, v_index) + offset) * unitUVNumber_inv;
}

RT_FUNCTION Sample_info Sample(const uint positionIndex, unsigned int &seed)
{
    uint UVN = unitUVNumber.x * unitUVNumber.y;
    Sample_info sample_info;

    float random_v = rnd(seed);
    uint left = 0;
    uint right = UVN - 1;
    float v = 0;
    float pdf = 0;
    uint index = 0;
    if (random_v < getPolicyQValue(positionIndex, 0)) {
        index = 0;
        pdf = getQCDF(positionIndex, 0);
    } else {
        while(true){
            uint center = (left + right) / 2;
            //v = getPolicyQValue(positionIndex, center);
            v = getQCDF(positionIndex, center);
            if(center == left){
                break;
            }
            if(v < random_v){
                left = center;
            } else {
                right = center;
            }
        }
        index = right;
        pdf = getQCDF(positionIndex, index) - getQCDF(positionIndex, index - 1);
    }

    float2 offset = make_float2(rnd(seed), rnd(seed));
    float2 uv = IndexToUV(index, offset);

    // TODO : change this!!
    float3 random_direction = UVToDirection(uv);

    pdf *= float(UVN);
    pdf *= 1/(4 * M_PIf);
    // sample_info.direction = UniformSampleSphere(rnd(seed), rnd(seed));
    sample_info.direction = random_direction;
    sample_info.pdf = pdf;
    return sample_info;
}

RT_FUNCTION Sample_info SampleNormalSensitive(const uint positionIndex, const float3 &normal, unsigned int &seed)
{
    uint UVN = unitUVNumber.x * unitUVNumber.y;
    Sample_info sample_info;

//    float2 normal_uv = DirectionToUV(normal);
//    uint normal_i = UVToIndex(normal_uv);
//    float q_value_sum = irradiance_table[make_uint2(normal_i, positionIndex)];
//    float3 new_normal = unitUVVectors[normal_i];

    float3 new_normal = normal;
    float q_value_sum = 0;
    for(unsigned int i=0; i<UVN; i++){
        float3 v_i = unitUVVectors[i];
        float cos_theta = dot(new_normal, v_i);
        if(cos_theta > 0.0f){
            float q_cos = getPolicyQValue(positionIndex, i) * cos_theta;
            q_value_sum += q_cos;
        }
    }

    float random_v = rnd(seed) * q_value_sum;
    float accumulated_value = 0;
    float current_value = 0;
    unsigned int index = 0;
    for(unsigned int i=0; i<UVN; i++){
        float3 v_i = unitUVVectors[i];
        float cos_theta = dot(new_normal, v_i);
        if(cos_theta > 0.0f){
            current_value = getPolicyQValue(positionIndex, i) * cos_theta;
            accumulated_value += current_value;
            if(accumulated_value >= random_v){
                index = i;
                break;
            }
        }
    }

    float2 offset = make_float2(rnd(seed), rnd(seed));
    float2 uv = IndexToUV(index, offset);
    float3 random_direction = UVToDirection(uv);

    float pdf = current_value / q_value_sum;
    pdf *= float(UVN);
    pdf *= 1/(4 * M_PIf);
    // sample_info.direction = UniformSampleSphere(rnd(seed), rnd(seed));
    sample_info.direction = random_direction;
    sample_info.pdf = pdf;
    return sample_info;
}

RT_FUNCTION Sample_info SampleNormalSensitiveReject(const uint positionIndex, const float3 &normal, const optix::Onb &onb, unsigned int &seed)
{
    uint UVN = unitUVNumber.x * unitUVNumber.y;
    Sample_info sample_info;

    float2 normal_uv = DirectionToUV(normal);
    uint normal_i = UVToIndex(normal_uv);
    const uint2 normal_index = make_uint2(normal_i, positionIndex);

    const float3& new_normal = normal;//unitUVVectors[normal_i];
    // float3 new_normal = normal;

    float q_value_sum = 0;
    float max_val = 0;
#if USE_MEMOIZATION
    if(irradiance_table[normal_index] > 0){
        q_value_sum = irradiance_table[normal_index];
        max_val = max_radiance_table[normal_index];
    }
    else {
#endif
    for(unsigned int i=0; i<UVN; i++){
        float cos_theta = dot(new_normal, unitUVVectors[i]);
        if(cos_theta > 0.0f){
            float q_cos = getPolicyQValue(positionIndex, i) * cos_theta;
            q_value_sum += q_cos;
            max_val = max(q_cos, max_val);
        }
    }
#if USE_MEMOIZATION
    irradiance_table[normal_index] = q_value_sum;
    max_radiance_table[normal_index] = max_val;
    }
#endif

    float3& direction = sample_info.direction;
    float& pdf = sample_info.pdf;

    float uniform = q_value_sum / float(UVN / 2);
    float c = uniform / max_val;
    float epsilon = c < 0.5 ? (1 - 2 * c) / (2 - 2 * c) : 0;
    max_val = (1-epsilon) * max_val + epsilon * uniform;

    while(true){
        direction = UniformSampleHemisphere(rnd(seed), rnd(seed));
        onb.inverse_transform(direction);
        const float2& uv = DirectionToUV(direction);
        uint directionIndex = UVToIndex(uv);
        pdf = getPolicyQValue(positionIndex, directionIndex) * max(0.0, dot(new_normal, direction));
        pdf = (1-epsilon) * pdf + epsilon * uniform;
        if(rnd(seed) < pdf / max_val){
            break;
        }
    }

    pdf = pdf / q_value_sum;
    pdf *= float(UVN);
    pdf *= 1/(4 * M_PIf);
    //sample_info.direction = direction;
    //sample_info.pdf = pdf;
    return sample_info;
}

RT_FUNCTION float Pdf(const uint positionIndex, const float3 &direction)
{
    const float2& uv = DirectionToUV(direction);
    uint directionIndex = UVToIndex(uv);
    float pdf;
    if (directionIndex == 0){
        pdf = getQCDF(positionIndex, 0);
    } else {
        pdf = getQCDF(positionIndex, directionIndex) - getQCDF(positionIndex, directionIndex - 1);
    }

    uint UVN = unitUVNumber.x * unitUVNumber.y;
    pdf *= float(UVN);
    pdf *= 1/(4 * M_PIf);
    return pdf;
}
}

