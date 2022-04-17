#include <optixu/optixu_math_namespace.h>
#include "optix/common/rt_function.h"

rtDeclareVariable(uint4,        unitCubeNumber, , );

namespace voxel
{
    RT_FUNCTION uint normalizedPositionToIndex(const float3 &p)
    {
        uint x = static_cast<unsigned int>(p.x * unitCubeNumber.x);
        uint y = static_cast<unsigned int>(p.y * unitCubeNumber.y);
        uint z = static_cast<unsigned int>(p.z * unitCubeNumber.z);
        //x = max(min(x, unitCubeNumber.x-1), 0);
        //y = max(min(y, unitCubeNumber.y-1), 0);
        //z = max(min(z, unitCubeNumber.z-1), 0);
        // return x+y+z;
        return x * unitCubeNumber.w + y * unitCubeNumber.z + z;

//        uint3 idx = make_uint3(p.x * unitCubeNumber.x, p.y * unitCubeNumber.y, p.z * unitCubeNumber.z);
//        idx = clamp(idx, make_uint3(0), unitCubeNumber - make_uint3(1));
//        uint idx_int = (idx.x * unitCubeNumber.y * unitCubeNumber.z) + (idx.y * unitCubeNumber.z) + idx.z;
//        return idx_int;
    }
}
