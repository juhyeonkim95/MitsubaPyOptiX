#pragma once

#include "optix/q_table/qTable.cuh"
#include "optix/common/rt_function.h"

using namespace optix;

// Non path-guided sampling
#define SAMPLE_UNIFORM 0
#define SAMPLE_BSDF 1
// Path-guided sampling
#define SAMPLE_INV 2
#define SAMPLE_REJ 3
#define SAMPLE_REJ_OPT 4
#define SAMPLE_MCMC 5
#define SAMPLE_SPHERICAL_INV 6
#define SAMPLE_QUADTREE 7

namespace radiance_record
{
RT_FUNCTION float Pdf(const float3 &position, const float3 &normal,
const float3 &direction, const unsigned int sample_type)
{
    switch(sample_type){
    case SAMPLE_QUADTREE: return getPdfQuadTree(position, direction);
    case SAMPLE_SPHERICAL_INV: return getPdfGrid(position, direction);
    }
    return 0.0f;
}

RT_FUNCTION Sample_info Sample(const float3 &position, const float3 &normal,
unsigned int &seed, const unsigned int sample_type)
{
    switch(sample_type){
    case SAMPLE_QUADTREE: return sampleScatteringDirectionProportionalToQQuadTree(position, normal, seed);
    case SAMPLE_SPHERICAL_INV: return sampleScatteringDirectionProportionalToQSphere(position, seed);
    }
    Sample_info sample_info;
    return sample_info;
}

}