#pragma once
#ifndef SHIRLEY_MAPPING_H
#define SHIRLEY_MAPPING_H

#include <optixu/optixu_math_namespace.h>
#include "optix/common/rt_function.h"

// "Peter Shirley, Notes on Adaptive Quadrature on the Hemisphere"

namespace shirley_mapping{
    RT_FUNCTION float3 UVToHemisphere(const float2 &uv)
    {
        float x = 2 * uv.x - 1;
        float y = 2 * uv.y - 1;

        float xx, yy, offset, theta, phi;
        if(y > -x){
            if(y<x){
                xx = x;
                if(y > 0){offset=0;yy=y;}
                else{offset=7;yy=x+y;}
            }else{
                xx = y;
                if(x > 0){offset=1;yy=y-x;}
                else{offset=2;yy=-x;}
            }
        }else {
            if(y>x){
                xx = -x;
                if(y > 0){offset=3;yy=-x-y;}
                else{offset=4;yy=-y;}
            }else{
                xx = -y;
                if(x > 0){offset=6;yy=x;}
                else{
                    if(y!=0){offset=5;yy=x-y;}
                    else{return make_float3(0,1,0);}
                }
            }
        }
        theta = acos(1 - xx*xx);
        phi = (M_PIf/4) * (offset + (yy / xx));
        return make_float3(sinf(theta) * cosf(phi), cosf(theta), -sinf(theta) * sinf(phi));
    }

    RT_FUNCTION float2 HemisphereToUV(const float3 &direction)
    {
        if(direction.x ==0 && direction.z == 0){return make_float2(0.5,0.5);}
        float Q_PIf = M_PIf * 0.25f;

        float theta = acos(abs(direction.y));
        float x = direction.x;
        float y = -direction.z;

        float phi = atan2(y, x);
        if (phi < 0){phi += (2 * M_PIf);}

        float xx = sqrt(1-cos(theta));
        uint offset = uint(phi / Q_PIf);
        float yy = phi / Q_PIf - float(offset);
        yy = yy * xx;
        float u, v;


        if(y > -x){
            if(y<x){
                u = xx;
                if(y > 0){v=yy;}
                else{v=yy-u;}
            }else{
                v = xx;
                if(x > 0){u=v-yy;}
                else{u=-yy;}
            }
        }else {
            if(y>x){
                u = -xx;
                if(y > 0){v=-u-yy;}
                else{v=-yy;}
            }else{
                v = -xx;
                if(x > 0){u=yy;}
                else{u=yy+v;}
            }
        }
        u = 0.5 * u + 0.5;
        v = 0.5 * v + 0.5;
        return make_float2(u, v);
    }

    RT_FUNCTION float3 UVToDirection(const float2 &uv){
        float2 uv_hemisphere = make_float2(uv.x >= 0.5 ? uv.x * 2 - 1: uv.x * 2, uv.y);
        float3 direction = UVToHemisphere(uv_hemisphere);
        direction.y = uv.x >= 0.5 ? -direction.y : direction.y;
        return direction;
    }

    RT_FUNCTION float2 DirectionToUV(const float3 &direction){
        float2 uv = HemisphereToUV(direction);
        uv.x = direction.y >= 0 ? uv.x * 0.5 : (uv.x + 1) * 0.5;
        return uv;
    }
}
#endif
