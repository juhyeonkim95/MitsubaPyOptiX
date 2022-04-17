#pragma once

#include <optixu/optixu_math_namespace.h>

#include "optix/bsdf/bsdf.h"
#include "optix/common/rt_function.h"
#include "optix/common/helpers.h"
#include "optix/common/prd_struct.h"
#include "optix/radiance_record/radiance_record.h"
#include "optix/bsdf/bsdf_flags.h"

rtDeclareVariable(float,         bsdf_sampling_fraction, , );

using namespace optix;

namespace mis_guided_sampling
{

RT_FUNCTION void Sample(
    const MaterialParameter& mat, const SurfaceInteraction &si,
    const optix::Onb &onb, unsigned int &seed, BSDFSample3f &bs)
{
    // ----------------------- Guided sampling ----------------------
    // Use Multiple Importance Sampling
    float bsdf_pdf = 0;
    float radiance_pdf = 0;

    // (1) BSDF Sampling (prop to f * cos)
    if (rnd(seed) <= bsdf_sampling_fraction){
        bsdf::Sample(mat, si, seed, bs);
        onb.inverse_transform(bs.wo);

        bsdf_pdf = bs.pdf;
        bs.weight *= bsdf_pdf;     // f * cos(theta) / pdf to f * cos(theta)

        radiance_pdf = radiance_record::Pdf(si.p, si.normal, bs.wo);
    }

    // (2) Radiance Sampling (prop to Q)
    else {
        Sample_info sample_info = radiance_record::Sample(si.p, si.normal, seed);

        bs.wo = sample_info.direction;
        radiance_pdf = sample_info.pdf;

        float3 wo_local = to_local(onb, bs.wo);
        bs.weight = bsdf::Eval(mat, si, wo_local);

        bsdf_pdf = bsdf::Pdf(mat, si, wo_local);
    }

    // Apply MIS
    bs.pdf = bsdf_sampling_fraction * bsdf_pdf + (1 - bsdf_sampling_fraction) * radiance_pdf;
    bs.pdf = max(bs.pdf, 1e-8);
    bs.weight /= bs.pdf;
    return;
}

RT_FUNCTION float Pdf(
    const MaterialParameter& mat, const SurfaceInteraction &si,
    const optix::Onb &onb, const float3 &wo)
{
    float3 wo_local = to_local(onb, wo);

    if(!(mat.bsdf_type | BSDFTypes::GuideTargets)){
        return bsdf::Pdf(mat, si, wo_local);
    }

    // ----------------------- Guided sampling ----------------------
    float radiance_pdf = radiance_record::Pdf(si.p, si.normal, wo);
    float bsdf_pdf = bsdf::Pdf(mat, si, wo_local);
    float pdf = bsdf_sampling_fraction * bsdf_pdf + (1 - bsdf_sampling_fraction) * radiance_pdf;
    return pdf;
}
}