#pragma once
#include <optixu/optixu_vector_types.h>
#include "optix/bsdf/bsdf_lobes.h"

using namespace optix;

struct BSDFSample3f
{
    float3 wo;
    float pdf;
    float eta;
    float3 weight;
    BSDFLobe sampledLobe;
};
