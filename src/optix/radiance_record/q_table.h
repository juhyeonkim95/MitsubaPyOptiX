#pragma once
#ifndef Q_TABLE_H
#define Q_TABLE_H

#include <optixu/optixu_math_namespace.h>
#include "optix/common/rt_function.h"
using namespace optix;

rtBuffer<float, 2>              q_table;
rtBuffer<float, 2>              q_table_accumulated;
rtBuffer<float, 2>              q_table_pdf;
rtBuffer<uint, 2>               q_table_visit_counts;
rtBuffer<uint, 2>               q_table_normal_counts;

RT_FUNCTION float getQCDF(unsigned int positionIndex, unsigned int i){
    return q_table_pdf[make_uint2(i, positionIndex)];
}

RT_FUNCTION float getPolicyQValue(unsigned int positionIndex, unsigned int i){
    return max(q_table_pdf[make_uint2(i, positionIndex)], 0.0f);
}

#endif
