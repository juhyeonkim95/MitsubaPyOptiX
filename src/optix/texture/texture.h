#pragma once

#include <optixu/optixu_math_namespace.h>
#include "optix/common/material_parameters.h"
#include "optix/common/prd_struct.h"


using namespace optix;

#define TEXTURE_CONSTANT 0
#define TEXTURE_BITMAP 1
#define TEXTURE_CHECKERBOARD 2

RT_FUNCTION float3 load_texture_rgb(const TextureParameter &texture, const SurfaceInteraction &si)
{
    float3 uv = texture.uv_transform * si.uv;

    switch(texture.type){
        case TEXTURE_BITMAP:
        {
            float4 texColor = optix::rtTex2D<float4>(texture.id , uv.x, 1 - uv.y);
            // to linear space --> already pre calculated
            // if(texture.srgb){
            //    texColor = make_float4(powf(texColor.x, 2.2f), powf(texColor.y, 2.2f), powf(texColor.z, 2.2f), texColor.w);
            // }
            return make_float3(texColor);
        }
	    case TEXTURE_CHECKERBOARD:
	    {
            float u = uv.x - int(uv.x);
            float v = uv.y - int(uv.y);
            if ((u > 0.5 && v > 0.5) || (u < 0.5 && v < 0.5)){
                return texture.color0;
            } else {
                return texture.color1;
            }
        }
    }
    return make_float3(.0,.0,.0);
}

RT_FUNCTION float load_texture_a(const TextureParameter &texture, const float3& texcoord)
{
    float3 uv = texture.uv_transform * texcoord;

    switch(texture.type){
        case TEXTURE_BITMAP:
        {
            float4 texColor = optix::rtTex2D<float4>(texture.id , uv.x, 1 - uv.y);
            return texColor.w;
        }
        case TEXTURE_CHECKERBOARD:
        {
            float u = uv.x - int(uv.x);
            float v = uv.y - int(uv.y);
            if ((u > 0.5 && v > 0.5) || (u < 0.5 && v < 0.5)){
                return texture.color0.x;
            } else {
                return texture.color1.x;
            }
        }
    }
    return 1.0;
}