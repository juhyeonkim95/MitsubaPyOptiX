#pragma once
#ifndef INTEGRATOR_COMMON_H
#define INTEGRATOR_COMMON_H
#include <optixu/optixu_math_namespace.h>
using namespace optix;
#include "optix/common/helpers.h"
#include "optix/common/rt_function.h"
#include "optix/app_config.h"
#include "optix/bsdf/bsdf.h"
#include "optix/light/light_sample.h"
#include "optix/light/light_pdf.h"

// scene geometry + material + lights
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(rtObject,      top_shadower, , );
rtBuffer<MaterialParameter> sysMaterialParameters;
rtBuffer<LightParameter> sysLightParameters;

// path tracer
rtDeclareVariable(float,         scene_epsilon, , );

rtDeclareVariable(unsigned int,  rr_begin_depth, , );
rtDeclareVariable(unsigned int,  max_depth, , );

rtDeclareVariable(unsigned int,  pathtrace_ray_type, , );
rtDeclareVariable(unsigned int,  pathtrace_shadow_ray_type, , );
#endif
