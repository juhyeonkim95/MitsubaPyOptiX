#pragma once
using namespace optix;
#include <optixu/optixu_math_namespace.h>

namespace warp
{
//-----------------------------------------------------------------------
RT_FUNCTION float cosine_sample_hemisphere_pdf(const float3& v)
//-----------------------------------------------------------------------
{
    return M_1_PIf * abs(v.z);
}

}
