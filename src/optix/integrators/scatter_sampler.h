#pragma once

#include <optixu/optixu_math_namespace.h>

#include "optix/bsdf/bsdf.h"
#include "optix/common/rt_function.h"
#include "optix/common/helpers.h"
#include "optix/common/prd_struct.h"
#include "optix/q_table/radiance_record.cuh"

rtDeclareVariable(float,         bsdf_sampling_fraction, , );
rtDeclareVariable(unsigned int,     sample_type, , );
rtDeclareVariable(unsigned int,     is_first_pass, , );

using namespace optix;

namespace scatter
{

RT_FUNCTION void Sample(
    const MaterialParameter& mat, const SurfaceInteraction &si,
    const optix::Onb &onb, unsigned int &seed, BSDFSample3f &bs)
{

    if(sample_type == SAMPLE_BSDF || is_first_pass){
        // ----------------------- BSDF sampling ----------------------
        bsdf::Sample(mat, si, seed, bs);
        onb.inverse_transform(bs.wo);
        return;
    }
    else if((sample_type == SAMPLE_QUADTREE) || (sample_type == SAMPLE_SPHERICAL_INV)){
        // ----------------------- Guided sampling ----------------------
        // Use Multiple Importance Sampling
        float bsdf_pdf;
        float radiance_pdf;

        // (1) BSDF Sampling (prop to f * cos)
        if (rnd(seed) < bsdf_sampling_fraction){
            bsdf::Sample(mat, si, seed, bs);
            onb.inverse_transform(bs.wo);

            bsdf_pdf = bs.pdf;
            bs.weight *= bsdf_pdf;     // f * cos(theta) / pdf to f * cos(theta)

            radiance_pdf = radiance_record::Pdf(si.p, si.normal, bs.wo, sample_type);
        }

        // (2) Radiance Sampling (prop to Q)
        else {
            Sample_info sample_info = radiance_record::Sample(si.p, si.normal, seed, sample_type);

            bs.wo = sample_info.direction;
            radiance_pdf = sample_info.pdf;

            float3 wo_local = to_local(onb, bs.wo);
            bs.weight = bsdf::Eval(mat, si, wo_local);

            bsdf_pdf = bsdf::Pdf(mat, si, wo_local);
        }
        if(length(bs.weight) == 0){
            return;
        }
        // Apply MIS
        bs.pdf = bsdf_sampling_fraction * bsdf_pdf + (1 - bsdf_sampling_fraction) * radiance_pdf;
        bs.weight /= bs.pdf;
        return;
    }
}

RT_FUNCTION void Pdf(
    const MaterialParameter& mat, const SurfaceInteraction &si,
    const optix::Onb &onb, const float3 &wo)
{

    if(sample_type == SAMPLE_BSDF || is_first_pass){
        // ----------------------- BSDF sampling ----------------------
        bsdf::Sample(mat, si, seed, bs);
        wo_local = to_local(onb, wo);
        return bsdf::Pdf(mat, si, wo_local);
    } else if((sample_type == SAMPLE_QUADTREE) || (sample_type == SAMPLE_SPHERICAL_INV)){
        // ----------------------- Guided sampling ----------------------
        float3 wo_local = to_local(onb, wo);

        float radiance_pdf = radiance_record::Pdf(si.p, si.normal, wo, sample_type);
        float bsdf_pdf = bsdf::Pdf(mat, si, wo_local);
        float pdf = bsdf_sampling_fraction * bsdf_pdf + (1 - bsdf_sampling_fraction) * radiance_pdf;
        return pdf;
    }
}
}
