#pragma once

#include <optixu/optixu_math_namespace.h>

#include "optix/bsdf/bsdf.h"
#include "optix/common/rt_function.h"
#include "optix/common/helpers.h"
#include "optix/common/prd_struct.h"
#include "optix/radiance_record/radiance_record.h"
#include "optix/bsdf/bsdf_flags.h"

using namespace optix;

namespace qcos_sampling
{

RT_FUNCTION void Sample(
    const MaterialParameter& mat, const SurfaceInteraction &si,
    const optix::Onb &onb, unsigned int &seed, BSDFSample3f &bs)
{
    // ----------------------- Guided sampling ----------------------

    Sample_info sample_info = radiance_record::SampleNormalSensitive(si.p, si.normal, seed);

    bs.wo = sample_info.direction;
    bs.pdf = sample_info.pdf;

    float3 wo_local = to_local(onb, bs.wo);
    bs.weight = bsdf::Eval(mat, si, wo_local);

    bs.pdf = max(bs.pdf, 1e-8);
    bs.weight /= bs.pdf;
    return;
}

}