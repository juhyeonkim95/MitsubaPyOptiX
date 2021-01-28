/*
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

using namespace optix;
static __host__ __device__ float3 mapUVToDirection(float2 uv)
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

static __host__ __device__ float3 mapThetaPhiToDirection(float2 thetaPhi)
{
    float theta = M_PIf * thetaPhi.x;
    float phi = 2 * M_PIf * thetaPhi.y;

    float3 direction = make_float3(sinf(theta) * cosf(phi), cosf(theta), -sinf(theta) * sinf(phi));
    return direction;
}

static __host__ __device__ float2 mapDirectionToUV(float3 direction)
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


static __host__ __device__ __inline__ float sampleSegment(double epsilon, float sigma, float smax) {
	return -log(1.0 - epsilon * (1.0 - exp(-sigma * smax))) / sigma;
}

static __host__ __device__ __inline__ float3 sampleHG(float g, float e1, float e2) {
    float s=2.0*e1-1.0, f = (1.0-g*g)/(1.0+g*s);
    float cost = 0.5*(1.0/g)*(1.0+g*g-f*f), sint = sqrtf(1.0-cost*cost);
    return make_float3(cosf(2.0f * M_PIf * e2) * sint, sinf(2.0f * M_PIf * e2) * sint, cost);
}
static __host__ __device__ __inline__ float3 sampleSphere(double e1, double e2) {
	double z = 1.0 - 2.0 * e1, sint = sqrtf(1.0 - z * z);
	return make_float3(cosf(2.0 * M_PIf * e2) * sint, sinf(2.0 * M_PIf * e2) * sint, z);
}

static __host__ __device__ __inline__ float HG_phase_function(float g, float cos_t){
    return (1-g*g)/(powf(max(0.01,1+g*g-2*g*cos_t),1.5));
}

static __host__ __device__ __inline__ void generateOrthoBasis(float3 &u, float3 &v, float3 w) {
	float3 coVec = w;
	if (fabs(w.x) <= fabs(w.y))
		if (fabs(w.x) <= fabs(w.z)) coVec = make_float3(0,-w.z,w.y);
		else coVec = make_float3(-w.y,w.x,0);
	else if (fabs(w.y) <= fabs(w.z)) coVec = make_float3(-w.z,0,w.x);
	else coVec = make_float3(-w.y,w.x,0);
	coVec = normalize(coVec);
	u = normalize(cross(w, coVec));
	v = cross(w, u);
}



static __device__ __inline__ float fresnel( float cos_theta_i, float cos_theta_t, float eta )
{
    const float rs = ( cos_theta_i - cos_theta_t*eta ) /
                     ( cos_theta_i + eta*cos_theta_t );
    const float rp = ( cos_theta_i*eta - cos_theta_t ) /
                     ( cos_theta_i*eta + cos_theta_t );

    return 0.5f * ( rs*rs + rp*rp );
}

static __host__ __device__ __inline__ float powerHeuristic(float a, float b)
{
	float t = a * a;
	return t / (b*b + t);
}


__device__ inline float4 ToneMap(const float4& c, float limit)
{
	float luminance = 0.3f*c.x + 0.6f*c.y + 0.1f*c.z;

	float4 col = c * 1.0f / (1.0f + luminance / limit);
	return make_float4(col.x, col.y, col.z, 1.0f);
}

__device__ inline float4 LinearToSrgb(const float4& c)
{
	const float kInvGamma = 1.0f / 2.2f;
	return make_float4(powf(c.x, kInvGamma), powf(c.y, kInvGamma), powf(c.z, kInvGamma), c.w);
}

__device__ float3 transform_point(Matrix4x4 &transformation, float3 v) {
	float4 a = make_float4(v.x, v.y, v.z, 1);
    a = transformation * a;
    return make_float3(a.x, a.y, a.z);
}

__device__ float3 transform_normal(Matrix4x4 &transformation, float3 v) {
    float4 a = make_float4(v.x, v.y, v.z, 0);
    a = transformation * a;
    return normalize(make_float3(a.x, a.y, a.z));
}