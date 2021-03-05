#pragma once
using namespace optix;
#include <optixu/optixu_math_namespace.h>

//----------------------------------------------------------------------
RT_FUNCTION float3 ImportanceSampleGTR1(float rgh, float r1, float r2)
//----------------------------------------------------------------------
{
    float a = max(0.001, rgh);
    float a2 = a * a;

    float phi = r1 * 2 * M_PIf;

    float cosTheta = sqrt((1.0 - pow(a2, 1.0 - r1)) / (1.0 - a2));
    float sinTheta = clamp(sqrtf(1.0 - (cosTheta * cosTheta)), 0.0f, 1.0f);
    float sinPhi = sin(phi);
    float cosPhi = cos(phi);

    return make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}

//----------------------------------------------------------------------
RT_FUNCTION float3 ImportanceSampleGTR2_aniso(float ax, float ay, float r1, float r2)
//----------------------------------------------------------------------
{
    float phi = r1 * 2 * M_PIf;

    float sinPhi = ay * sin(phi);
    float cosPhi = ax * cos(phi);
    float tanTheta = sqrt(r2 / (1 - r2));

    return make_float3(tanTheta * cosPhi, tanTheta * sinPhi, 1.0);
}

//----------------------------------------------------------------------
RT_FUNCTION float3 ImportanceSampleGTR2(float rgh, float r1, float r2)
//----------------------------------------------------------------------
{
    float a = max(0.001, rgh);

    float phi = r1 * 2 * M_PIf;

    float cosTheta = sqrt((1.0 - r2) / (1.0 + (a * a - 1.0) * r2));
    float sinTheta = clamp(sqrtf(1.0 - (cosTheta * cosTheta)), 0.0f, 1.0f);
    float sinPhi = sin(phi);
    float cosPhi = cos(phi);

    return make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}

//-----------------------------------------------------------------------
RT_FUNCTION float SchlickFresnel(float u)
//-----------------------------------------------------------------------
{
    float m = clamp(1.0f - u, 0.0f, 1.0f);
    float m2 = m * m;
    return m2 * m2 * m; // pow(m,5)
}

//-----------------------------------------------------------------------
RT_FUNCTION float DielectricFresnel(float cos_theta_i, float eta)
//-----------------------------------------------------------------------
{
    float sinThetaTSq = eta * eta * (1.0f - cos_theta_i * cos_theta_i);

    // Total internal reflection
    if (sinThetaTSq > 1.0)
        return 1.0;

    float cos_theta_t = sqrt(max(1.0 - sinThetaTSq, 0.0));

//    float rs = (eta * cos_theta_t - cos_theta_i) / (eta * cos_theta_t + cos_theta_i);
//    float rp = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    float rs = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    float rp = (eta * cos_theta_t - cos_theta_i) / (eta * cos_theta_t + cos_theta_i);

    return 0.5f * (rs * rs + rp * rp);
}

//-----------------------------------------------------------------------
RT_FUNCTION float GTR1(float NDotH, float a)
//-----------------------------------------------------------------------
{
    if (a >= 1.0)
        return (1.0 / M_PIf);
    float a2 = a * a;
    float t = 1.0 + (a2 - 1.0) * NDotH * NDotH;
    return (a2 - 1.0) / (M_PIf * log(a2) * t);
}

//-----------------------------------------------------------------------
RT_FUNCTION float GTR2(float NDotH, float a)
//-----------------------------------------------------------------------
{
    float a2 = a * a;
    float t = 1.0 + (a2 - 1.0) * NDotH * NDotH;
    return a2 / (M_PIf * t * t);
}

//-----------------------------------------------------------------------
RT_FUNCTION float GTR2_aniso(float NDotH, float HDotX, float HDotY, float ax, float ay)
//-----------------------------------------------------------------------
{
    float a = HDotX / ax;
    float b = HDotY / ay;
    float c = a * a + b * b + NDotH * NDotH;
    return 1.0 / (M_PIf * ax * ay * c * c);
}

//-----------------------------------------------------------------------
RT_FUNCTION float SmithG_GGX(float NDotV, float alphaG)
//-----------------------------------------------------------------------
{
    float a = alphaG * alphaG;
    float b = NDotV * NDotV;
    return 1.0 / (NDotV + sqrt(a + b - a * b));
}

//-----------------------------------------------------------------------
RT_FUNCTION float G1(float NDotV, float HDotV, float alphaG)
//-----------------------------------------------------------------------
{
    if(NDotV * HDotV < 0 ){
        return 0.0f;
    }
    float alphaSq  = alphaG * alphaG;
    float cosThetaSq = NDotV * NDotV;
    float tanThetaSq = max(1.0f - cosThetaSq, 0.0f)/cosThetaSq;
    return 2.0 / (1 + sqrt(1 + alphaSq *tanThetaSq));
}

//-----------------------------------------------------------------------
RT_FUNCTION float SmithG_GGX_aniso(float NDotV, float VDotX, float VDotY, float ax, float ay)
//-----------------------------------------------------------------------
{
    float a = VDotX * ax;
    float b = VDotY * ay;
    float c = NDotV;
    return 1.0 / (NDotV + sqrt(a * a + b * b + c * c));
}

//-----------------------------------------------------------------------
RT_FUNCTION float3 CosineSampleHemisphere(float r1, float r2)
//-----------------------------------------------------------------------
{
    float3 dir;
    float r = sqrt(r1);
    float phi = 2 * M_PIf * r2;
    dir.x = r * cos(phi);
    dir.y = r * sin(phi);
    dir.z = sqrt(max(0.0, 1.0 - dir.x * dir.x - dir.y * dir.y));

    return dir;
}

RT_FUNCTION float3 sqrt3f(float3 &v)
{
    return make_float3(sqrt(v.x), sqrt(v.y), sqrt(v.z));
}

////-----------------------------------------------------------------------
//RT_FUNCTION float3 UniformSampleHemisphere(float r1, float r2)
////-----------------------------------------------------------------------
//{
//    float r = sqrt(max(0.0, 1.0 - r1 * r1));
//    float phi = 2 * M_PIf * r2;
//
//    return make_float3(r * cos(phi), r * sin(phi), r1);
//}
//
////-----------------------------------------------------------------------
//RT_FUNCTION float3 UniformSampleSphere(float r1, float r2)
////-----------------------------------------------------------------------
//{
//    float z = 1.0 - 2.0 * r1;
//    float r = sqrt(max(0.0, 1.0 - z * z));
//    float phi = 2 * M_PIf * r2;
//
//    return make_float3(r * cos(phi), r * sin(phi), z);
//}