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
    int hitTriIdx;
};

struct PerRayData_pathtrace_shadow
{
    bool inShadow;
    unsigned int seed;
};

struct Sample_info
{
    float3 direction;
    float pdf;
};

// Explicitly not named Onb to not conflict with the optix::Onb
// Tangent-Bitangent-Normal orthonormal space.
struct TBN
{
  // Default constructor to be able to include it into other structures when needed.
  RT_FUNCTION TBN()
  {
  }

  RT_FUNCTION TBN(const optix::float3& n)
  : normal(n)
  {
    if (fabsf(normal.z) < fabsf(normal.x))
    {
      tangent.x =  normal.z;
      tangent.y =  0.0f;
      tangent.z = -normal.x;
    }
    else
    {
      tangent.x =  0.0f;
      tangent.y =  normal.z;
      tangent.z = -normal.y;
    }
    tangent   = optix::normalize(tangent);
    bitangent = optix::cross(normal, tangent);
  }

  // Constructor for cases where tangent, bitangent, and normal are given as ortho-normal basis.
  RT_FUNCTION TBN(const optix::float3& t, const optix::float3& b, const optix::float3& n)
  : tangent(t)
  , bitangent(b)
  , normal(n)
  {
  }

  // Normal is kept, tangent and bitangent are calculated.
  // Normal must be normalized.
  // Must not be used with degenerated vectors!
  RT_FUNCTION TBN(const optix::float3& tangent_reference, const optix::float3& n)
  : normal(n)
  {
    bitangent = optix::normalize(optix::cross(normal, tangent_reference));
    tangent   = optix::cross(bitangent, normal);
  }

  RT_FUNCTION void negate()
  {
    tangent   = -tangent;
    bitangent = -bitangent;
    normal    = -normal;
  }


  RT_FUNCTION optix::float3 to_local(optix::float3& p) const
  {
    return optix::make_float3(optix::dot(p, tangent),
                          optix::dot(p, bitangent),
                          optix::dot(p, normal));
  }

  RT_FUNCTION optix::float3 to_world(optix::float3& p) const
  {
    return p.x * tangent + p.y * bitangent + p.z * normal;
  }

  optix::float3 tangent;
  optix::float3 bitangent;
  optix::float3 normal;
};

struct SurfaceInteraction
{
    float3 p;
    float3 wi;
    float3 uv;
    float3 emission;
    float3 normal;
    // float3 tangent;
    // float3 bitangent;

    int material_id;
    int light_id;
    bool is_valid;
    int hitTriIdx;
    float t;
    unsigned int seed;

    // TBN onb;

//    RT_FUNCTION void set_from_normal(const optix::float3& n)
//    {
//        normal = n;
//        if (fabsf(normal.z) < fabsf(normal.x))
//        {
//            tangent.x =  normal.z;
//            tangent.y =  0.0f;
//            tangent.z = -normal.x;
//        }
//        else
//        {
//            tangent.x =  0.0f;
//            tangent.y =  normal.z;
//            tangent.z = -normal.y;
//        }
//        tangent   = optix::normalize(tangent);
//        bitangent = optix::cross(normal, tangent);
//    }
//    RT_FUNCTION optix::float3 to_local(optix::float3& p) const
//    {
//        return optix::make_float3(optix::dot(p, tangent),
//                          optix::dot(p, bitangent),
//                          optix::dot(p, normal));
//    }
//
//    RT_FUNCTION optix::float3 to_world(optix::float3& p) const
//    {
//        return p.x * tangent + p.y * bitangent + p.z * normal;
//    }

};