#include <optixu/optixu_math_namespace.h>
#include "optix/common/rt_function.h"

rtDeclareVariable(uint3,        unitCubeNumber, , );

namespace voxel
{
    RT_FUNCTION uint normalizedPositionToIndex(const float3 &p)
    {
        uint3 idx = make_uint3(p.x * unitCubeNumber.x, p.y * unitCubeNumber.y, p.z * unitCubeNumber.z);
        idx = clamp(idx, make_uint3(0), unitCubeNumber - make_uint3(1));
        uint idx_int = (idx.x * unitCubeNumber.y * unitCubeNumber.z) + (idx.y * unitCubeNumber.z) + idx.z;
        return idx_int;
    }
}
