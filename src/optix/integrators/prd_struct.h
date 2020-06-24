#pragma once

#include <optixu/optixu_vector_types.h>

using namespace optix;

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
    float t;
    float3 current_attenuation;
};

struct PerRayData_pathtrace_shadow
{
    bool inShadow;
};

struct Sample_info
{
    float3 direction;
    float p_w;
};