#pragma once

#include <optixu/optixu_math_namespace.h>
using namespace optix;
static __host__ __device__ __inline__ float UniformConePdf(float cos_theta_max)
{
    return 1 / (2 * M_PIf * (1 - cos_theta_max));
}

static __host__ __device__ __inline__ float3 UniformSampleHemisphere(float u1, float u2)
{
    float sin_theta = sqrtf(max(0.0, 1.0 - u1 * u1));
    float phi = 2.0 * M_PIf * u2;
	return make_float3(cosf(phi) * sin_theta, sinf(phi) * sin_theta, u1);
}

static __host__ __device__ __inline__ float2 UniformSampleTriangle(float u1, float u2)
{
    float z = sqrtf(u1);
    return make_float2(1-z, u2*z);
}

static __host__ __device__ __inline__ float3 UniformSampleSphere(float u1, float u2)
{
    float z = 1 - 2 * u1;
    float sin_theta = sqrtf(max(0.0, 1.0 - z * z));
    float phi = 2.0 * M_PIf * u2;
	return make_float3(cosf(phi) * sin_theta, sinf(phi) * sin_theta, z);
}

static __host__ __device__ __inline__ float3 UniformSampleSphereGivenP(const float3& p, const float3& p_c, float radius, float u1, float u2, float&pdf)
{
    float3 L = p - p_c;
    float Ldist2 = dot(L,L);
    float Ldist = sqrtf(Ldist2);
    float3 Lnorm = L / Ldist;

    float cos_theta_max = radius / Ldist;
    float cos_theta = (1 - u1) + u1 * cos_theta_max;
    float sin_theta = sqrt(max(0.0, 1.0-cos_theta * cos_theta));
    float phi = 2.0 * M_PIf * u2;

    float3 dir = make_float3(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);
    optix::Onb onb( Lnorm );
    onb.inverse_transform( dir );

    float area = (2 * M_PIf * (1 - cos_theta_max)) * radius * radius;
    pdf = 1 / area;
    return dir;
}

static __host__ __device__ __inline__ float2 UniformSampleDisk(float u1, float u2)
{
    float r = sqrtf(u1);
    float theta = 2 * M_PIf * u2;
    return make_float2(r, theta);
}
