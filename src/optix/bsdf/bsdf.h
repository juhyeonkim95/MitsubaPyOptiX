#pragma once

#include "optix/bsdf/diffuse.h"
#include "optix/bsdf/dielectric.h"
#include "optix/bsdf/roughdielectric.h"
#include "optix/bsdf/conductor.h"
#include "optix/bsdf/roughconductor.h"
#include "optix/bsdf/plastic.h"
#include "optix/bsdf/roughplastic.h"
#include "optix/bsdf/bsdf_flags.h"
//#include "optix/bsdf/disney2.h"


using namespace optix;


namespace bsdf{
// Sample for wo using brdf
// returns direction with f * cos(theta) / pdf
RT_CALLABLE_PROGRAM void Sample(
    const MaterialParameter &mat, const SurfaceInteraction &si,
    unsigned int &seed, BSDFSample3f &bs
)
{
    switch(mat.bsdf_type){
    case BSDFTypes::Diffuse: diffuse::Sample(mat, si, seed, bs); return;
    case BSDFTypes::Dielectric: dielectric::Sample(mat, si, seed, bs); return;
    case BSDFTypes::RoughDielectric: roughdielectric::Sample(mat, si, seed, bs); return;
    case BSDFTypes::Conductor: conductor::Sample(mat, si, seed, bs); return;
    case BSDFTypes::RoughConductor: roughconductor::Sample(mat, si, seed, bs); return;
    case BSDFTypes::Plastic: plastic::Sample(mat, si, seed, bs); return;
    case BSDFTypes::RoughPlastic: roughplastic::Sample(mat, si, seed, bs); return;
    }
    diffuse::Sample(mat, si, seed, bs);return;
}


// Evaluate pdf for wo and wi
RT_CALLABLE_PROGRAM float Pdf(const MaterialParameter &mat, const SurfaceInteraction &si, const float3 &wo)
{
    switch(mat.bsdf_type){
    case BSDFTypes::Diffuse: return diffuse::Pdf(mat, si, wo);
    case BSDFTypes::Dielectric: return dielectric::Pdf(mat, si, wo);
    case BSDFTypes::RoughDielectric: return roughdielectric::Pdf(mat, si, wo);
    case BSDFTypes::Conductor: return conductor::Pdf(mat, si, wo);
    case BSDFTypes::RoughConductor: return roughconductor::Pdf(mat, si, wo);
    case BSDFTypes::Plastic: return plastic::Pdf(mat, si, wo);
    case BSDFTypes::RoughPlastic: return roughplastic::Pdf(mat, si, wo);
    default: return 0;
    }
}

// Evaluate brdf * cos value for wo and wi.
RT_CALLABLE_PROGRAM float3 Eval(const MaterialParameter &mat, const SurfaceInteraction &si, const float3 &wo)
{
    switch(mat.bsdf_type){
    case BSDFTypes::Diffuse:  return diffuse::Eval(mat, si, wo);
    case BSDFTypes::Dielectric: return dielectric::Eval(mat, si, wo);
    case BSDFTypes::RoughDielectric: return roughdielectric::Eval(mat, si, wo);
    case BSDFTypes::Conductor: return conductor::Eval(mat, si, wo);
    case BSDFTypes::RoughConductor: return roughconductor::Eval(mat, si, wo);
    case BSDFTypes::Plastic: return plastic::Eval(mat, si, wo);
    case BSDFTypes::RoughPlastic: return roughplastic::Eval(mat, si, wo);
    default: return make_float3(0.0f);
    }
}
}
