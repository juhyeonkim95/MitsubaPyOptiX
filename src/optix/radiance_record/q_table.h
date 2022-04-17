#pragma once
#ifndef Q_TABLE_H
#define Q_TABLE_H

#include <optixu/optixu_math_namespace.h>
#include "optix/common/rt_function.h"
using namespace optix;

rtBuffer<float, 2>              q_table;
rtBuffer<float, 2>              q_table_accumulated;
rtBuffer<float, 2>              q_table_pdf;
rtBuffer<float, 2>              q_table_cdf;
rtBuffer<uint, 2>               q_table_visit_counts;
rtBuffer<uint, 2>               q_table_normal_counts;
rtBuffer<float, 2>              irradiance_table;
rtBuffer<float, 2>              max_radiance_table;

RT_FUNCTION float getQCDF(unsigned int positionIndex, unsigned int i){
    return q_table_cdf[make_uint2(i, positionIndex)];
}

RT_FUNCTION float getPolicyQValue(unsigned int positionIndex, unsigned int i){
    return q_table_pdf[make_uint2(i, positionIndex)];
}


#endif
