#pragma once

#include <optixu/optixu_math_namespace.h>
#include "optix/sampling/mis_guided_sampling.h"
#include "optix/sampling/qcos_sampling.h"
#include "optix/sampling/rejection_sampling.h"

rtDeclareVariable(unsigned int,     sampling_strategy, , );

#define SAMPLE_Q_COS_PROPORTION 3
#define SAMPLE_Q_QUADTREE 7

namespace guided_sampling
{
RT_FUNCTION void Sample(
const MaterialParameter& mat, const SurfaceInteraction &si,
const optix::Onb &onb, unsigned int &seed, BSDFSample3f &bs)
{
    // Not target of guided sampling
    if(!(mat.bsdf_type & BSDFTypes::GuideTargets)){
        bsdf::Sample(mat, si, seed, bs);
        onb.inverse_transform(bs.wo);
        return;
    }
    mis_guided_sampling::Sample(mat, si, onb, seed, bs);
//#if SAMPLING_STRATEGY == SAMPLE_MIS
//    mis_guided_sampling::Sample(mat, si, onb, seed, bs);
//#elif SAMPLING_STRATEGY == SAMPLE_Q_COS_INVERSION
//    qcos_sampling::Sample(mat, si, onb, seed, bs);
//#elif SAMPLING_STRATEGY == SAMPLE_Q_COS_REJECT_MIX
//    rejection_sampling::Sample(mat, si, onb, seed, bs);
//#else
//    mis_guided_sampling::Sample(mat, si, onb, seed, bs);
//#endif
}

RT_FUNCTION float Pdf(
    const MaterialParameter& mat, const SurfaceInteraction &si,
    const optix::Onb &onb, const float3 &wo)
{
    return 0.0;
}

}
