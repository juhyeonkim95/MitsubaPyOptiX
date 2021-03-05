/*
 Copyright Disney Enterprises, Inc.  All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License
 and the following modification to it: Section 6 Trademarks.
 deleted and replaced with:

 6. Trademarks. This License does not grant permission to use the
 trade names, trademarks, service marks, or product names of the
 Licensor and its affiliates, except as required for reproducing
 the content of the NOTICE file.

 You may obtain a copy of the License at
 http://www.apache.org/licenses/LICENSE-2.0
*/
#pragma once

#include <optixu/optixu_math_namespace.h>
#include "optix/common/random.h"
#include "optix/common/sampling.h"
#include "optix/common/material_parameters.h"
#include "optix/bsdf/disney_sampling.h"

using namespace optix;
namespace disney
{

//-----------------------------------------------------------------------
RT_FUNCTION float3 EvalDielectricReflection(State &state, float3 &V, float3 &N, float3 &L, float3 &H, float &pdf)
//-----------------------------------------------------------------------
{
    if (dot(N, L) < 0.0) return make_float3(0.0);

    float F = DielectricFresnel(dot(V, H), state.eta);
    float D = GTR2(dot(N, H), state.mat.roughness);

    pdf = D * dot(N, H) * F / (4.0 * dot(V, H));

    float G = SmithG_GGX(abs(dot(N, L)), state.mat.roughness) * SmithG_GGX(dot(N, V), state.mat.roughness);
    return state.mat.albedo * F * D * G;
}

//-----------------------------------------------------------------------
RT_FUNCTION float3 EvalDielectricRefraction(State &state, float3 &V, float3 &N, float3 &L, float3 &H, float &pdf)
//-----------------------------------------------------------------------
{

    float F = DielectricFresnel(abs(dot(V, H)), state.eta);
    float D = GTR2(dot(N, H), state.mat.roughness);

    // before
    //float denomSqrt = dot(L, H) * state.eta + dot(V, H);
    // after
    float denomSqrt = dot(L, H) + dot(V, H) * state.eta;
    pdf = D * dot(N, H) * (1.0 - F) * abs(dot(L, H)) / (denomSqrt * denomSqrt);

    float G = SmithG_GGX(abs(dot(N, L)), state.mat.roughness) * SmithG_GGX(dot(N, V), state.mat.roughness);
    // before
    return state.mat.albedo * (1.0 - F) * D * G * abs(dot(V, H)) * abs(dot(L, H)) * 4.0 * state.eta * state.eta / (denomSqrt * denomSqrt);
    // after
    // return state.mat.albedo * (1.0 - F) * D * G * abs(dot(V, H)) * abs(dot(L, H)) * 4.0  / (denomSqrt * denomSqrt);

}

//-----------------------------------------------------------------------
RT_FUNCTION float3 EvalSpecular(State &state, float3 &Cspec0, float3 &V, float3 &N, float3 &L, float3 &H, float &pdf)
//-----------------------------------------------------------------------
{
    if (dot(N, L) < 0.0) return make_float3(0.0f);

    float D = GTR2_aniso(dot(N, H), dot(H, state.tangent), dot(H, state.bitangent), state.mat.ax, state.mat.ay);
    pdf = D * dot(N, H) / (4.0 * dot(V, H));

    float FH = SchlickFresnel(dot(L, H));
    float3 F = lerp(Cspec0, make_float3(1.0f), FH);
    float G = SmithG_GGX_aniso(dot(N, L), dot(L, state.tangent), dot(L, state.bitangent), state.mat.ax, state.mat.ay);
    G *= SmithG_GGX_aniso(dot(N, V), dot(V, state.tangent), dot(V, state.bitangent), state.mat.ax, state.mat.ay);
    return F * D * G;
}

//-----------------------------------------------------------------------
RT_FUNCTION float3 EvalClearcoat(State &state, float3 &V, float3 &N, float3 &L, float3 &H, float &pdf)
//-----------------------------------------------------------------------
{
    if (dot(N, L) < 0.0) return make_float3(0.0);

    float D = GTR1(dot(N, H), state.mat.clearcoatRoughness);
    pdf = D * dot(N, H) / (4.0 * dot(V, H));

    float FH = SchlickFresnel(dot(L, H));
    float F = lerp(0.04, 1.0, FH);
    float G = SmithG_GGX(dot(N, L), 0.25) * SmithG_GGX(dot(N, V), 0.25);
    return make_float3(0.25 * state.mat.clearcoat * F * D * G);
}

//-----------------------------------------------------------------------
RT_FUNCTION float3 EvalDiffuse(State &state, float3 &Csheen, float3 &V, float3 &N, float3 &L, float3 &H, float &pdf)
//-----------------------------------------------------------------------
{
    if (dot(N, L) < 0.0) return make_float3(0.0);

    pdf = dot(N, L) * (1.0 / M_PIf);

    float FL = SchlickFresnel(dot(N, L));
    float FV = SchlickFresnel(dot(N, V));
    float FH = SchlickFresnel(dot(L, H));
    float Fd90 = 0.5 + 2.0 * dot(L, H) * dot(L, H) * state.mat.roughness;
    float Fd = lerp(1.0, Fd90, FL) * lerp(1.0, Fd90, FV);
    float3 Fsheen = FH * state.mat.sheen * Csheen;
    return ((1.0 / M_PIf) * Fd * (1.0 - state.mat.subsurface) * state.mat.albedo + Fsheen) * (1.0 - state.mat.metallic);
}

//-----------------------------------------------------------------------
RT_FUNCTION float3 EvalSubsurface(State &state, float3 &V, float3 &N, float3 &L, float &pdf)
//-----------------------------------------------------------------------
{
    pdf = (1.0 / (2 * M_PIf));

    float FL = SchlickFresnel(abs(dot(N, L)));
    float FV = SchlickFresnel(dot(N, V));
    float Fd = (1.0f - 0.5f * FL) * (1.0f - 0.5f * FV);
    return sqrt3f(state.mat.albedo) * state.mat.subsurface * (1.0 / M_PIf) * Fd * (1.0 - state.mat.metallic) * (1.0 - state.mat.transmission);
}


//-----------------------------------------------------------------------
RT_CALLABLE_PROGRAM float3 DisneySample(State &state, float3 &V, float3 &N, unsigned int& seed, float3 &L, float& pdf)
//-----------------------------------------------------------------------
{
    state.isSubsurface = false;
    pdf = 0.0;
    float3 f = make_float3(0.0);

    float r1 = rnd(seed);
    float r2 = rnd(seed);

    float diffuseRatio = 0.5 * (1.0 - state.mat.metallic);
    float transWeight = (1.0 - state.mat.metallic) * state.mat.transmission;

    float3 Cdlin = state.mat.albedo;
    float Cdlum = 0.3 * Cdlin.x + 0.6 * Cdlin.y + 0.1 * Cdlin.z; // luminance approx.

    float3 Ctint = Cdlum > 0.0 ? Cdlin / Cdlum : make_float3(1.0f); // normalize lum. to isolate hue+sat
    float3 Cspec0 = lerp(state.mat.specular * 0.08 * lerp(make_float3(1.0), Ctint, state.mat.specularTint), Cdlin, state.mat.metallic);
    float3 Csheen = lerp(make_float3(1.0), Ctint, state.mat.sheenTint);

    // BSDF
    if (rnd(seed) < transWeight)
    {
        float3 H = ImportanceSampleGTR2(state.mat.roughness, r1, r2);
        H = normalize(H);
        H = state.tangent * H.x + state.bitangent * H.y + N * H.z;
        //float3 H = N;

        float3 R = reflect(-V, H);
        float F = DielectricFresnel(abs(dot(V, H)), state.eta);
        pdf = 1.0;
        // Reflection/Total internal reflection
        if (rnd(seed) < F)
        {
            L = normalize(R);
            f = EvalDielectricReflection(state, V, N, L, H, pdf);
        }
        else // Transmission
        {
            // optix::refract takes ior
            optix::refract(L, -V, H, 1.0 / state.eta);
            f = EvalDielectricRefraction(state, V, N, L, H, pdf);
        }
//        float G = microfacet::G(L, V, N, state.mat.roughness);
//        // float G = SmithG_GGX(abs(dot(N, wi)), mat.roughness) * SmithG_GGX(dot(N, wo), mat.roughness) * 4 * abs(dot(N, wi)) * dot(N, wo);
//
//        float D = microfacet::D(dot(H, N), state.mat.roughness);
//        float pm = microfacet::pdf(H, N, state.mat.roughness);
//        float3 f = state.mat.albedo * abs(dot(V, H)) * G * D / abs(pm * dot(V, N));
//        return f;

        //float G = SmithG_GGX(abs(dot(N, L)), state.mat.roughness) * SmithG_GGX(dot(N, V), state.mat.roughness);
        //G = G * 4 * abs(dot(N, L)) * dot(N, V);
        //f = state.mat.albedo * abs(dot(V, H) * G) / abs(dot(N, H) * dot(N, V)* dot(N, L));
        //f = state.mat.albedo * G * 4 * abs(dot(V, H)) / abs(dot(N, H) * dot(N, L));
        f *= transWeight;
        pdf *= transWeight;
    }
    else // BRDF
    {
        if (rnd(seed) < diffuseRatio)
        {
            // Diffuse transmission. A way to approximate subsurface scattering
            if (rnd(seed) < state.mat.subsurface)
            {
                L = UniformSampleHemisphere(r1, r2);

                L = state.tangent * L.x + state.bitangent * L.y - N * L.z;

                f = EvalSubsurface(state, V, N, L, pdf);
                pdf *= state.mat.subsurface * diffuseRatio;

                state.isSubsurface = true; // Required when sampling lights from inside surface
            }
            else // Diffuse
            {
                L = CosineSampleHemisphere(r1, r2);
                L = state.tangent * L.x + state.bitangent * L.y + N * L.z;

                float3 H = normalize(L + V);

                f = EvalDiffuse(state, Csheen, V, N, L, H, pdf);
                pdf *= (1.0 - state.mat.subsurface) * diffuseRatio;
            }
        }
        else // Specular
        {
            float primarySpecRatio = 1.0 / (1.0 + state.mat.clearcoat);

            // Sample primary specular lobe
            if (rnd(seed) < primarySpecRatio)
            {
                // TODO: Implement http://jcgt.org/published/0007/04/01/
                float3 H = ImportanceSampleGTR2_aniso(state.mat.ax, state.mat.ay, r1, r2);
                H = state.tangent * H.x + state.bitangent * H.y + N * H.z;
                L = normalize(reflect(-V, H));

                f = EvalSpecular(state, Cspec0, V, N, L, H, pdf);
                pdf *= primarySpecRatio * (1.0 - diffuseRatio);
            }
            else // Sample clearcoat lobe
            {
                float3 H = ImportanceSampleGTR1(state.mat.clearcoatRoughness, r1, r2);
                H = state.tangent * H.x + state.bitangent * H.y + N * H.z;
                L = normalize(reflect(-V, H));

                f = EvalClearcoat(state, V, N, L, H, pdf);
                pdf *= (1.0 - primarySpecRatio) * (1.0 - diffuseRatio);
            }
        }

        f *= (1.0 - transWeight);
        pdf *= (1.0 - transWeight);
    }
    return f;
}

//-----------------------------------------------------------------------
RT_CALLABLE_PROGRAM float3 DisneyEval(State &state, float3 &V, float3 &N, float3 &L, float& pdf)
//-----------------------------------------------------------------------
{
    float3 H;

    if (dot(N, L) < 0.0)
        H = normalize(L * (1.0 / state.eta) + V);
    else
        H = normalize(L + V);

    if (dot(N, H) < 0.0)
        H = -H;

    float diffuseRatio = 0.5 * (1.0 - state.mat.metallic);
    float primarySpecRatio = 1.0 / (1.0 + state.mat.clearcoat);
    float transWeight = (1.0 - state.mat.metallic) * state.mat.transmission;

    float3 brdf = make_float3(0.0);
    float3 bsdf = make_float3(0.0);
    float brdfPdf = 0.0;
    float bsdfPdf = 0.0;

    // BSDF
    if (transWeight > 0.0)
    {
        // Transmission
        if (dot(N, L) < 0.0)
        {
            bsdf = EvalDielectricRefraction(state, V, N, L, H, bsdfPdf);
        }
        else // Reflection
        {
            bsdf = EvalDielectricReflection(state, V, N, L, H, bsdfPdf);
        }
    }

    float m_pdf;

    if (transWeight < 1.0)
    {
        // Subsurface
        if (dot(N, L) < 0.0)
        {
            // TODO: Double check this. Fails furnace test when used with rough transmission
            if (state.mat.subsurface > 0.0)
            {
                brdf = EvalSubsurface(state, V, N, L, m_pdf);
                brdfPdf = m_pdf * state.mat.subsurface * diffuseRatio;
            }
        }
        // BRDF
        else
        {
            float3 Cdlin = state.mat.albedo;
            float Cdlum = 0.3 * Cdlin.x + 0.6 * Cdlin.y + 0.1 * Cdlin.z; // luminance approx.

            float3 Ctint = Cdlum > 0.0 ? Cdlin / Cdlum : make_float3(1.0f); // normalize lum. to isolate hue+sat
            float3 Cspec0 = lerp(state.mat.specular * 0.08 * lerp(make_float3(1.0), Ctint, state.mat.specularTint), Cdlin, state.mat.metallic);
            float3 Csheen = lerp(make_float3(1.0), Ctint, state.mat.sheenTint);

            // Diffuse
            brdf += EvalDiffuse(state, Csheen, V, N, L, H, m_pdf);
            brdfPdf += m_pdf * (1.0 - state.mat.subsurface) * diffuseRatio;

            // Specular
            brdf += EvalSpecular(state, Cspec0, V, N, L, H, m_pdf);
            brdfPdf += m_pdf * primarySpecRatio * (1.0 - diffuseRatio);

            // Clearcoat
            brdf += EvalClearcoat(state, V, N, L, H, m_pdf);
            brdfPdf += m_pdf * (1.0 - primarySpecRatio) * (1.0 - diffuseRatio);
        }
    }

    pdf = lerp(brdfPdf, bsdfPdf, transWeight);
    return lerp(brdf, bsdf, transWeight);
}

}