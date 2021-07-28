#pragma once

#ifndef MATERIAL_PARAMETER_H
#define MATERIAL_PARAMETER_H

#include "optix/common/rt_function.h"

#include <optixu/optixu_matrix_namespace.h>
using namespace optix;

struct TextureParameter
{
    unsigned int type;
    unsigned int id;
    optix::Matrix3x3 uv_transform;
    optix::float3 color0;
    optix::float3 color1;
    unsigned int srgb;
};

struct MaterialParameter
{
//	RT_FUNCTION MaterialParameter()
//	{
//		albedo = optix::make_float3(1.0f, 1.0f, 1.0f);
//		emission = optix::make_float3(0.0f);
//		metallic = 0.0;
//		subsurface = 0.0f;
//		specular = 0.5f;
//		roughness = 0.5f;
//		specularTint = 0.0f;
//		anisotropic = 0.0f;
//		sheen = 0.0f;
//		sheenTint = 0.5f;
//		clearcoat = 0.0f;
//		clearcoatRoughness = 1.0f;
//		albedoID = RT_TEXTURE_ID_NULL;
//		opacity = 0.5f;
//        isTwosided = 0;
//        intIOR = 1.504;
//        extIOR = 1.0;
//	}
    unsigned int bsdf_type;
    // albedo or albedo texture
	float3 albedo;
	int albedo_texture_id;

    float3 diffuse_reflectance;
    int diffuse_reflectance_texture_id;
    float3 specular_reflectance;
    int specular_reflectance_texture_id;
    float3 specular_transmittance;
    int specular_transmittance_texture_id;
    float alpha;
    int alpha_texture_id;
    float opacity;
    int opacity_texture_id;

	optix::float3 emission;
	float metallic;
	float subsurface;
	float specular;
	float roughness;
	float specularTint;
	float anisotropic;
	float sheen;
	float sheenTint;
	float clearcoat;
	float clearcoatRoughness;
	unsigned int isTwosided;
	float intIOR;
	float extIOR;
	float transmission;
	float ax;
	float ay;
	int distribution_type;
	int nonlinear;
	unsigned int bumpID;
    optix::float3 eta;
    optix::float3 k;
};

struct State
{
    MaterialParameter mat;
    float eta;
    float3 bitangent;
    float3 tangent;
    float3 normal;
    float3 ffnormal;
    bool isSubsurface;
};

#endif
