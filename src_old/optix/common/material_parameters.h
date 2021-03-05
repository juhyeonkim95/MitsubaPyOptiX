#pragma once

#ifndef MATERIAL_PARAMETER_H
#define MATERIAL_PARAMETER_H

#include "optix/common/rt_function.h"

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

	int albedoID;
	optix::float3 albedo;
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
	float opacity;
	unsigned int isTwosided;
	float intIOR;
	float extIOR;
	float transmission;
	float ax;
	float ay;
	int distribution_type;
	int nonlinear;
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
