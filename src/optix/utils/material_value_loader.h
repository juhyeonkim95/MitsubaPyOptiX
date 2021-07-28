#pragma once

#include "optix/common/rt_function.h"
#include "optix/common/material_parameters.h"
#include "optix/texture/texture.h"
#include "optix/common/prd_struct.h"

using namespace optix;

rtBuffer<TextureParameter> sysTextureParameters;

//RT_FUNCTION float3 eval_albedo(const MaterialParameter &mat, const float3 &texcoord){
//    if(mat.albedo_texture_id >= 0){
//        return load_texture_rgb(sysTextureParameters[mat.albedo_texture_id], texcoord);
//    } else {
//        return mat.albedo;
//    }
//}

RT_FUNCTION float3 eval_diffuse_reflectance(const MaterialParameter &mat, const SurfaceInteraction &si){
    if(mat.diffuse_reflectance_texture_id >= 0){
        return load_texture_rgb(sysTextureParameters[mat.diffuse_reflectance_texture_id], si);
    } else {
        return mat.diffuse_reflectance;
    }
}

RT_FUNCTION float3 eval_specular_reflectance(const MaterialParameter &mat, const SurfaceInteraction &si){
    if(mat.specular_reflectance_texture_id >= 0){
        return load_texture_rgb(sysTextureParameters[mat.specular_reflectance_texture_id], si);
    } else {
        return mat.specular_reflectance;
    }
}

RT_FUNCTION float3 eval_specular_transmittance(const MaterialParameter &mat, const SurfaceInteraction &si){
    if(mat.specular_transmittance_texture_id >= 0){
        return load_texture_rgb(sysTextureParameters[mat.specular_transmittance_texture_id], si);
    } else {
        return mat.specular_transmittance;
    }
}

RT_FUNCTION float eval_roughness(const MaterialParameter &mat, const SurfaceInteraction &si){
    if(mat.alpha_texture_id >= 0){
        return load_texture_a(sysTextureParameters[mat.alpha_texture_id], si.uv);
    } else {
        return mat.alpha;
    }
}

RT_FUNCTION float eval_opacity(const MaterialParameter &mat, const float3 &texcoord){
    if(mat.opacity_texture_id >= 0){
        return load_texture_a(sysTextureParameters[mat.opacity_texture_id], texcoord);
    } else {
        return mat.opacity;
    }
}
