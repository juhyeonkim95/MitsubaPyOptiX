#include <optixu/optixu_math_namespace.h>
#include "optix/radiance_record/q_table.h"
#include "optix/radiance_record/directional_mapping/directional_mapping.h"

using namespace optix;
rtDeclareVariable(uint2,        unitUVNumber, , );

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
    // not required!!
    unsigned int u_index = (dir_index / unitUVNumber.y);
    unsigned int v_index = (dir_index % unitUVNumber.y);

    return make_float2(u_index + offset.x, v_index + offset.y);
}

RT_FUNCTION Sample_info Sample(const uint positionIndex, unsigned int &seed)
{
    uint UVN = unitUVNumber.x * unitUVNumber.y;
    Sample_info sample_info;

    float random_v = rnd(seed);
    uint left = 0;
    uint right = UVN * 2 - 1;
    float v = 0;
    float pdf = 0;
    uint index = 0;
    if (random_v < getPolicyQValue(positionIndex, 0)) {
        index = 0;
        pdf = getQCDF(positionIndex, 0);
    } else {
        while(true){
            uint center = (left + right) / 2;
            v = getPolicyQValue(positionIndex, center);
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

    pdf *= float(2 * UVN);
    pdf *= 1/(4 * M_PIf);
    sample_info.direction = random_direction;
    sample_info.pdf = pdf;
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
    pdf *= float(2 * UVN);
    pdf *= 1/(4 * M_PIf);
    return pdf;
}
}

