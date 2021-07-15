#pragma once

#include <optixu/optixu_vector_types.h>
#include "optix/common/material_parameters.h"

using namespace optix;

//
//struct MaterialParameter
//{
//    float3 diffuse_color;
//};

struct PerRayData_pathtrace
{
    float3 result;
    float3 radiance;
    float3 attenuation;
    float3 origin;
    float3 direction;
    float3 normal;
    float3 diffuse_color;
    unsigned int seed;
    int depth;
    int countEmitted;
    int done;
    bool isMissed;
    float t;
    float3 current_attenuation;
    float scatterPdf;
    bool volume_scattered;
    float3 wo;
    float3 bsdfDir;
    float pdf;
    bool isSpecular;
    int material_type;
    int valid_scatter_count;
    int invalid_scatter_count;
};

struct PerRayData_pathtrace_shadow
{
    bool inShadow;
};

struct Sample_info
{
    float3 direction;
    float pdf;
};


struct SurfaceInteraction3f
{
    float3 p;
    float3 n;
    float3 wi;
};

//struct ParallelogramLight
//{
//    optix::float3 corner;
//    optix::float3 v1, v2;
//    optix::float3 normal;
//    optix::float3 emission;
//};