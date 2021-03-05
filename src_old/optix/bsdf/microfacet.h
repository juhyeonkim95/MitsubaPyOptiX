#pragma once
using namespace optix;
#include <optixu/optixu_math_namespace.h>
enum DistributionEnum
{
    Beckmann = 0,
    Phong = 1,
    GGX = 2
};

namespace microfacet
{
//-----------------------------------------------------------------------
RT_FUNCTION float roughnessToAlpha(DistributionEnum dist, float roughness)
//-----------------------------------------------------------------------
{
    float MinAlpha = 1e-3f;
    roughness = max(roughness, MinAlpha);

    if (dist == Phong)
        return 2.0f/(roughness*roughness) - 2.0f;
    else
        return roughness;
}

//-----------------------------------------------------------------------
RT_FUNCTION float D(DistributionEnum dist, float alpha, const float3& m)
//-----------------------------------------------------------------------
{
    if(m.z <= 0.0f){
        return 0.0f;
    }
    switch(dist){
    case Beckmann: {
        float alphaSq = alpha*alpha;
        float cosThetaSq = m.z*m.z;
        float tanThetaSq = max(1.0f - cosThetaSq, 0.0f)/cosThetaSq;
        float cosThetaQu = cosThetaSq*cosThetaSq;
        return M_1_PIf*exp(-tanThetaSq/alphaSq)/(alphaSq*cosThetaQu);
    }
    case Phong:
        return (alpha + 2.0f)*M_1_PIf * 0.5f *pow(m.z, alpha);
    case GGX:{
        float a2 = alpha * alpha;
        float t = 1.0 + (a2 - 1.0) * m.z * m.z;
        return a2 / (M_PIf * t * t);
    }
    }
    return 0.0f;
}

//-----------------------------------------------------------------------
RT_FUNCTION float G1(DistributionEnum dist, float alpha, const float3 &v, const float3 &m)
//-----------------------------------------------------------------------
{
    if (dot(v, m)*v.z <= 0.0f)
        return 0.0f;
    switch(dist){
    case Beckmann: {
        float cosThetaSq = v.z*v.z;
        float tanTheta = abs(sqrt(max(1.0f - cosThetaSq, 0.0f))/v.z);
        float a = 1.0f/(alpha*tanTheta);
        if (a < 1.6f)
            return (3.535f*a + 2.181f*a*a)/(1.0f + 2.276f*a + 2.577f*a*a);
        else
            return 1.0f;
    }
    case Phong:{
        float cosThetaSq = v.z*v.z;
        float tanTheta = abs(sqrt(max(1.0f - cosThetaSq, 0.0f))/v.z);
        float a = std::sqrt(0.5f*alpha + 1.0f)/tanTheta;
        if (a < 1.6f)
            return (3.535f*a + 2.181f*a*a)/(1.0f + 2.276f*a + 2.577f*a*a);
        else
            return 1.0f;
    }
    case GGX:{
        float alphaSq = alpha*alpha;
        float cosThetaSq = v.z*v.z;
        float tanThetaSq = max(1.0f - cosThetaSq, 0.0f)/cosThetaSq;
        return 2.0f/(1.0f + sqrt(1.0f + alphaSq*tanThetaSq));
    }
    }

    return 0.0f;
    //float a = alphaG * alphaG;
    //float b = NDotV * NDotV;
    //return 2 * NDotV / (NDotV + sqrt(a + b - a * b));
}

//-----------------------------------------------------------------------
RT_FUNCTION float G(DistributionEnum dist, float alpha, const float3& wi, const float3& wo, const float3& m)
//-----------------------------------------------------------------------
{
    return G1(dist, alpha, wi, m) * G1(dist, alpha, wo, m);
}

//-----------------------------------------------------------------------
RT_FUNCTION float pdf(DistributionEnum dist, float alpha, const float3 &m)
//-----------------------------------------------------------------------
{
    return D(dist, alpha, m) * m.z;
}

//----------------------------------------------------------------------
RT_FUNCTION float3 sample(DistributionEnum dist, float alpha, float r1, float r2)
//----------------------------------------------------------------------
{
    float phi = r2 * 2 * M_PIf;
    float cosTheta = 0.0f;

    switch(dist){
    case Beckmann:{
        float tanThetaSq = -alpha*alpha*log(1.0f - r1);
        cosTheta = 1.0f/std::sqrt(1.0f + tanThetaSq);
        break;
    }
    case Phong:{
        cosTheta = float(pow(r1, 1.0/(alpha + 2.0)));
        break;
    }
    case GGX:{
        float tanThetaSq = alpha*alpha*r1/(1.0f - r1);
        cosTheta = 1.0f/sqrt(1.0f + tanThetaSq);
        break;
    }
    }
    float r = sqrtf(max(1.0f - cosTheta*cosTheta, 0.0f));
    return make_float3(cosf(phi)*r, sinf(phi)*r, cosTheta);

//    float a = max(0.001, rgh);
//    float cosTheta = sqrt((1.0 - r2) / (1.0 + (a * a - 1.0) * r2));
//    float sinTheta = clamp(sqrtf(1.0 - (cosTheta * cosTheta)), 0.0f, 1.0f);
//    float sinPhi = sin(phi);
//    float cosPhi = cos(phi);
//    return make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}
}