#include <optixu/optixu_math_namespace.h>
#include "optix/common/helpers.h"
#include "optix/common/prd_struct.h"
#include "optix/common/random.h"
#include "optix/common/sampling.h"
#include "optix/q_table/data_structure.cuh"
#include "optix/common/rt_function.h"
// #include "optix/bsdf/bsdf.h"


using namespace optix;

rtBuffer<float, 2>              q_table;
rtBuffer<float, 2>              irradiance_table;
rtBuffer<float, 2>              max_radiance_table;

rtBuffer<float, 2>              q_table_accumulated;
rtBuffer<float, 2>              q_table_pdf;
rtBuffer<uint, 2>               q_table_visit_counts;
rtBuffer<uint, 2>               q_table_normal_counts;
rtBuffer<unsigned int>              invalid_sample_counts;
rtBuffer<unsigned int>              valid_sample_counts;

rtBuffer<float2, 2>              mcmc_table;



rtBuffer<float3, 1>             unitUVVectors;


rtDeclareVariable(unsigned int,     q_table_update_method, , );
rtDeclareVariable(unsigned int,     use_memoization, ,);


static __host__ __device__ float3 getDirectionFrom(uint index, float2 offset)
{
    unsigned int u_index = (index / unitUVNumber.y);
    unsigned int v_index = (index % unitUVNumber.y);
    bool inverted = false;
    if (u_index > unitUVNumber.x){
        u_index -= unitUVNumber.x;
        inverted = true;
    }
    float u_index_r = (float(u_index) + offset.x)/(float(unitUVNumber.x));
    float v_index_r = (float(v_index) + offset.y)/(float(unitUVNumber.y));
    float3 random_direction = mapUVToDirection(make_float2(u_index_r, v_index_r));
    if (inverted){
        random_direction.y *= -1;
    }
    return random_direction;
}


static __host__ __device__ float getQValue(float3 position, float3 direction)
{
    uint positionIndex = positionToIndex(position);
    uint rayIndex = directionToIndex(positionIndex, direction);
    return q_table[make_uint2(rayIndex, positionIndex)];
}

static __host__ __device__ float getQValueIndex(uint positionIndex, uint rayIndex)
{
    return q_table[make_uint2(rayIndex, positionIndex)];
}

static __host__ __device__ void setQValue(float3 position, float3 direction, float value)
{
    uint positionIndex = positionToIndex(position);
    uint rayIndex = directionToIndex(positionIndex, direction);
    q_table[make_uint2(rayIndex, positionIndex)] = value;
}

static __host__ __device__ uint2 getPosDirIndex(const float3 &position, const float3 &direction)
{
    uint positionIndex = positionToIndex(position);
    uint rayIndex = directionToIndex(positionIndex, direction);
    uint2 idx = make_uint2(rayIndex, positionIndex);
    return idx;
}

static __host__ __device__ void accumulateNormalValue(float3 position, float3 normal)
{
    uint positionIndex = positionToIndex(position);
    uint rayIndex = directionToIndex(positionIndex, normal);
    uint2 idx = make_uint2(rayIndex, positionIndex);
    atomicAdd(&q_table_normal_counts[idx], 1);
}

static __host__ __device__ void increment_invalid_sample(float3 &position)
{
    uint positionIndex = positionToIndex(position);
    atomicAdd(&invalid_sample_counts[positionIndex], 1);
}

static __host__ __device__ void increment_valid_sample(float3 &position)
{
    uint positionIndex = positionToIndex(position);
    atomicAdd(&valid_sample_counts[positionIndex], 1);
}



static __host__ __device__ uint updateVisit(float3 position, float3 direction)
{
    uint positionIndex = positionToIndex(position);
    uint rayIndex = directionToIndex(positionIndex, direction);
    uint2 idx = make_uint2(rayIndex, positionIndex);
    float v = q_table_visit_counts[idx];
    atomicAdd(&q_table_visit_counts[idx], 1);
    //q_table_visit_counts[idx] = v + 1;

    return v;
}

static __host__ __device__ uint getVisit(float3 position, float3 direction)
{
    uint positionIndex = positionToIndex(position);
    uint rayIndex = directionToIndex(positionIndex, direction);
    uint2 idx = make_uint2(rayIndex, positionIndex);
    float v = q_table_visit_counts[idx];
    return v;
}



static __host__ __device__ float2 mutate1(float2& x, unsigned int &seed){
    return make_float2(rnd(seed), rnd(seed));
}

static __host__ __device__ float2 mutate2(float2& x, unsigned int &seed){
    float2 x_ = x + 0.1 * (make_float2(rnd(seed), rnd(seed)) - 0.5);
    if(x_.x > 1){
        x_.x -=1;
    } else if(x_.x < 0){
        x_.x +=1;
    }
    if(x_.y > 1){
        x_.y -=1;
    } else if(x_.y < 0){
        x_.y +=1;
    }
    return x_;
}

static __host__ __device__ float2 mutate(float2& x, unsigned int &seed){
    if(rnd(seed) > 0.1){
        return mutate2(x, seed);
    } else {
        return mutate1(x, seed);
    }
}

static __host__ __device__ Sample_info sampleScatteringDirectionProportionalToQMCMC(float3 position, float3 normal, unsigned int &seed)
{
    uint UVN = unitUVNumber.x * unitUVNumber.y;
    uint positionIndex = positionToIndex(position);
    uint n_i = directionToIndex(positionIndex, normal);
    float q_value_sum = 0;
    if(irradiance_table[make_uint2(n_i, positionIndex)] > 0.0){
        q_value_sum = irradiance_table[make_uint2(n_i, positionIndex)];
    } else {
        for(unsigned int i=0; i<UVN * 2; i++){
            float3 v_i = unitUVVectors[i];//getDirectionFrom(i, make_float2(0.5, 0.5));
            float cos_theta = dot(normal, v_i);
            if(cos_theta > 0.0f){
                float c = cos_theta;//considerCosineTerm?cos_theta:1.0;
                q_value_sum += getPolicyQValue(positionIndex, i) * c;
            }
        }
        irradiance_table[make_uint2(n_i, positionIndex)] = q_value_sum;
    }

    Sample_info sample_info;

    float2 x1 = mcmc_table[make_uint2(n_i, positionIndex)];
    float2 x2 = mutate(x1, seed);

    optix::Onb onb( normal );
    float3 v1 = UniformSampleHemisphere(x1.x, x1.y);
    onb.inverse_transform(v1);
    float3 v2 = UniformSampleHemisphere(x2.x, x2.y);
    onb.inverse_transform(v2);
    //float3 v1 = mapThetaPhiToDirection(x1);
    //float3 v2 = mapThetaPhiToDirection(x2);

    uint uv1 = directionToIndex(positionIndex, v1);
    uint uv2 = directionToIndex(positionIndex, v2);

    float q1 = getPolicyQValue(positionIndex, uv1) * max(0.0, dot(normal, v1));
    float q2 = getPolicyQValue(positionIndex, uv2) * max(0.0, dot(normal, v2));
    float pdf;
    q1 += 1e-7;
    q2 += 1e-7;

    float a = min(1.0, q2 / q1);
    float3 direction;
    if(rnd(seed) < a){
        x1 = x2;
        mcmc_table[make_uint2(n_i, positionIndex)] = x2;
        direction = v2;
        pdf = q2;
    } else{
        direction = getDirectionFrom(uv1, make_float2(rnd(seed), rnd(seed)));
        pdf = q1;
    }
    pdf /= q_value_sum;
    pdf *= float(UVN);
    pdf *= 1/(2*M_PIf);
    // pdf = max(pdf, 0.01f);

    sample_info.direction = direction;
    sample_info.pdf = pdf;
    return sample_info;
}

static __host__ __device__ Sample_info sampleScatteringDirectionProportionalToQReject(float3 position, float3 normal, unsigned int &seed)
{
    uint UVN = unitUVNumber.x * unitUVNumber.y;
    uint positionIndex = positionToIndex(position);
    uint n_i = directionToIndex(positionIndex, normal);
    optix::Onb onb( normal );
    float q_value_sum = 0;
    float max_val = 0;

    if(use_memoization && irradiance_table[make_uint2(n_i, positionIndex)] > 0.0){
        q_value_sum = irradiance_table[make_uint2(n_i, positionIndex)];
        max_val = max_radiance_table[make_uint2(n_i, positionIndex)];
    } else {
        for(unsigned int i=0; i<UVN * 2; i++){
            float3 v_i = unitUVVectors[i];//getDirectionFrom(i, make_float2(0.5, 0.5));
            float cos_theta = dot(normal, v_i);
            if(cos_theta > 0.0f){
                float c = cos_theta;//considerCosineTerm?cos_theta:1.0;
                float q_cos = getPolicyQValue(positionIndex, i) * c;
                if(q_cos > max_val){
                    max_val = q_cos;
                }
                q_value_sum += q_cos;
            }
        }
        if (use_memoization){
            irradiance_table[make_uint2(n_i, positionIndex)] = q_value_sum;
            max_radiance_table[make_uint2(n_i, positionIndex)] = max_val;
        }
    }

    Sample_info sample_info;
    float3 direction;
    float pdf;

    while(true){
        float2 x = make_float2(rnd(seed), rnd(seed));
        direction = UniformSampleHemisphere(x.x, x.y);
        onb.inverse_transform(direction);
        float e = rnd(seed);
        uint uv = directionToIndex(positionIndex, direction);
        pdf = getPolicyQValue(positionIndex, uv) * max(0.0, dot(normal, direction));
        if(e < pdf / max_val){
            break;
        }
    }

    pdf /= q_value_sum;
    pdf *= float(UVN);
    pdf *= 1/(2*M_PIf);

    sample_info.direction = direction;
    sample_info.pdf = pdf;
    return sample_info;
}

static __host__ __device__ Sample_info sampleScatteringDirectionProportionalToQReject2(float3 position, float3 normal, unsigned int &seed)
{
    uint UVN = unitUVNumber.x * unitUVNumber.y;
    uint positionIndex = positionToIndex(position);
    uint n_i = directionToIndex(positionIndex, normal);
    optix::Onb onb( normal );
    float q_value_sum = 0;
    float max_val = 0;
    if(use_memoization && irradiance_table[make_uint2(n_i, positionIndex)] > 0.0){
        q_value_sum = irradiance_table[make_uint2(n_i, positionIndex)];
        max_val = max_radiance_table[make_uint2(n_i, positionIndex)];
    } else {
        for(unsigned int i=0; i<UVN * 2; i++){
            float3 v_i = unitUVVectors[i];//getDirectionFrom(i, make_float2(0.5, 0.5));
            float cos_theta = dot(normal, v_i);
            if(cos_theta > 0.0f){
                float c = cos_theta;//considerCosineTerm?cos_theta:1.0;
                float q_cos = getPolicyQValue(positionIndex, i) * c;
                if(q_cos > max_val){
                    max_val = q_cos;
                }
                q_value_sum += q_cos;
            }
        }
        if (use_memoization){
            irradiance_table[make_uint2(n_i, positionIndex)] = q_value_sum;
            max_radiance_table[make_uint2(n_i, positionIndex)] = max_val;
        }
    }

    Sample_info sample_info;
    float3 direction;
    float pdf;

    float uniform = q_value_sum / float(UVN);
    float c = uniform / max_val;
    float epsilon = c < 0.5 ? (1 - 2 * c) / (2 - 2 * c) : 0;
    max_val = (1-epsilon) * max_val + epsilon * uniform;

    while(true){
        float2 x = make_float2(rnd(seed), rnd(seed));
        direction = UniformSampleHemisphere(x.x, x.y);
        onb.inverse_transform(direction);
        float e = rnd(seed);
        uint uv = directionToIndex(positionIndex, direction);
        pdf = getPolicyQValue(positionIndex, uv) * max(0.0, dot(normal, direction));
        pdf = (1-epsilon) * pdf + epsilon * uniform;

        if(e < pdf / max_val){
            break;
        }
    }

    pdf /= q_value_sum;
    pdf *= float(UVN);
    pdf *= 1/(2*M_PIf);
    //pdf = max(pdf, 0.01f);

    sample_info.direction = direction;
    sample_info.pdf = pdf;
    return sample_info;
}

static __host__ __device__ Sample_info sampleScatteringDirectionProportionalToQ(float3 position, float3 normal, bool considerCosineTerm, unsigned int &seed)
{
    uint positionIndex = positionToIndex(position);
    uint UVN = unitUVNumber.x * unitUVNumber.y;
    Sample_info sample_info;
    float q_value_sum = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        float3 v_i = unitUVVectors[i];
        //float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
        float cos_theta = dot(normal, v_i);
        if(cos_theta > 0.0f){
            float c = considerCosineTerm?cos_theta:1.0;
            q_value_sum += getPolicyQValue(positionIndex, i) * c;
        }
    }

    float random_v = rnd(seed) * q_value_sum;
    float accumulated_value = 0;
    unsigned int index = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        float3 v_i = unitUVVectors[i];
        //float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
        float cos_theta = dot(normal, v_i);
        if(cos_theta > 0.0f){
            float c = considerCosineTerm?cos_theta:1.0;
            accumulated_value += getPolicyQValue(positionIndex, i) * c;
            if(accumulated_value >= random_v){
                index = i;
                break;
            }
        }
    }

    //rtPrintf("attenuation %f \n",q_value_sum);
    float3 random_direction;
    while(true){
        random_direction = getDirectionFrom(index, make_float2(rnd(seed), rnd(seed)));
        if(dot(random_direction, normal) > 0.0f){
            break;
        }
    }

    float3 v_i = getDirectionFrom(index, make_float2(0.5, 0.5));
    float cos_theta = dot(normal, v_i);
    float c = considerCosineTerm?cos_theta:1.0;

    float pdf = (getPolicyQValue(positionIndex, index) * c) / q_value_sum;
    pdf *= float(UVN);
    pdf *= 1/(2*M_PIf);
    sample_info.direction = random_direction;
    sample_info.pdf = pdf;
    return sample_info;
}


static __host__ __device__ Sample_info sampleScatteringDirectionMaximumQ(float3 position, float3 normal, unsigned int &seed)
{
    uint positionIndex = positionToIndex(position);
    uint UVN = unitUVNumber.x * unitUVNumber.y;
    Sample_info sample_info;

    float q_value_sum = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        //float3 v_i = unitUVVectors[i];
        float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
        if(dot(normal, v_i) > 0.0f){
            q_value_sum += getPolicyQValue(positionIndex, i);//#q_table_pdf[make_uint2(positionIndex, i)];
        }
    }

    float q_value_max = 0;
    unsigned int index = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        //float3 v_i = unitUVVectors[i];
        float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
        if(dot(normal, v_i) > 0.0f){
            float v = getPolicyQValue(positionIndex, i);
            if(q_value_max < v){
                q_value_max = v;
                index = i;
            }
        }
    }

    //rtPrintf("attenuation %f \n",q_value_sum);
    float3 random_direction;
    while(true){
        random_direction = getDirectionFrom(index, make_float2(rnd(seed), rnd(seed)));
        if(dot(random_direction, normal) > 0.0f){
            break;
        }
    }

    float p_w = getPolicyQValue(positionIndex, index) / q_value_sum;
    float pdf = p_w * UVN * ( 1 / (2 * M_PIf));
    pdf = clamp(pdf, 0.01f, 1.0f);

    sample_info.direction = random_direction;
    sample_info.pdf = pdf;
    return sample_info;
}

static __host__ __device__ float getExpectedQValue(float3 &position, float3 &normal, float3 &wi)
{
    uint positionIndex = positionToIndex(position);
    uint UVN = unitUVNumber.x * unitUVNumber.y;

    //optix::Onb onb( normal );
    //float3 wi_local = to_local(onb, wi);

    float q_value_sum = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        float3 wo = unitUVVectors[i];
        float cos_theta = max(0.0, dot(normal, wo));
        q_value_sum += cos_theta > 0.0f ? getQValueIndex(positionIndex, i) * cos_theta : 0;
    }
    float expected_q_value = 2 * q_value_sum / float(UVN);
    return expected_q_value;
}

static __host__ __device__ float getExpectedQValueVolume(float g, float3 &position, float3 &wo)
{
    uint positionIndex = positionToIndex(position);
    uint UVN = unitUVNumber.x * unitUVNumber.y;

    float q_value_sum = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        float3 wi = getDirectionFrom(i, make_float2(0.5, 0.5));

        float q_next = q_table[make_uint2(i, positionIndex)];
        // float3 f = Eval(mat, normal, wo, wi);
        float f = HG_phase_function(g, dot(wi,wo));
        //float cos_theta = max(0.0, dot(normal, wi));
        //float c = cos_theta;
        q_value_sum += q_next * f;
    }
    float expected_q_value = q_value_sum / (2 * float(UVN));
    return expected_q_value;
}

static __host__ __device__ float getMaximumQValue(float3 position, float3 normal, float3 wo)
{
    uint positionIndex = positionToIndex(position);
    uint UVN = unitUVNumber.x * unitUVNumber.y;

    float q_value_max = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
        float v = q_table[make_uint2(i, positionIndex)];
        if(max(0.0, dot(normal, v_i)) > 0.0 && v > q_value_max){
            q_value_max = v;
        }
    }
    return q_value_max;
}

static __host__ __device__ float getMaximumQValueVolume(float g, float3 position, float3 wo)
{
    uint positionIndex = positionToIndex(position);
    uint UVN = unitUVNumber.x * unitUVNumber.y;

    float q_value_max = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
        float v = q_table[make_uint2(i, positionIndex)];
        if(v > q_value_max){
            q_value_max = v;
        }
    }
    return q_value_max;
}

static __host__ __device__ Sample_info sampleScatteringDirectionProportionalToQVolume(float3 position, unsigned int &seed)
{
    uint positionIndex = positionToIndex(position);
    uint UVN = unitUVNumber.x * unitUVNumber.y;
    Sample_info sample_info;
    float q_value_sum = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        //float3 v_i = unitUVVectors[i];
        float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
        q_value_sum += getPolicyQValue(positionIndex, i);
    }

    float random_v = rnd(seed) * q_value_sum;
    float accumulated_value = 0;
    unsigned int index = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        //float3 v_i = unitUVVectors[i];
        float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
        accumulated_value += getPolicyQValue(positionIndex, i);
        if(accumulated_value >= random_v){
            index = i;
            break;
        }
    }
    float3 random_direction;
    random_direction = getDirectionFrom(index, make_float2(rnd(seed), rnd(seed)));

    float3 v_i = getDirectionFrom(index, make_float2(0.5, 0.5));

    float pdf = (getPolicyQValue(positionIndex, index)) / q_value_sum * 2 * float(UVN);
    pdf = max(pdf, 0.1f);

    sample_info.direction = random_direction;
    sample_info.pdf = pdf;
    return sample_info;
}

static __host__ __device__ Sample_info sampleScatteringDirectionProportionalToQVolumeHG(float3 position, float3 ray_direction, float g, unsigned int &seed)
{
    uint positionIndex = positionToIndex(position);
    uint UVN = unitUVNumber.x * unitUVNumber.y;
    Sample_info sample_info;
    float q_value_sum = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        //float3 v_i = unitUVVectors[i];
        float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
        float hg = HG_phase_function(g, dot(ray_direction, v_i));
        q_value_sum += getPolicyQValue(positionIndex, i) * hg;
    }

    float random_v = rnd(seed) * q_value_sum;
    float accumulated_value = 0;
    unsigned int index = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
        //float3 v_i = unitUVVectors[i];
        float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
        float hg = HG_phase_function(g, dot(ray_direction, v_i));
        accumulated_value += getPolicyQValue(positionIndex, i) * hg;
        if(accumulated_value >= random_v){
            index = i;
            break;
        }
    }
    float3 random_direction;
    random_direction = getDirectionFrom(index, make_float2(rnd(seed), rnd(seed)));

    float3 v_i = getDirectionFrom(index, make_float2(0.5, 0.5));
    float hg = HG_phase_function(g, dot(ray_direction, v_i));
    float pdf = (getPolicyQValue(positionIndex, index)) * hg / q_value_sum * 2 * UVN;
    pdf = max(pdf, 0.1f);

    // float p_w = (getPolicyQValue(positionIndex, index) * hg) / q_value_sum;

    sample_info.direction = random_direction;
    sample_info.pdf = pdf;
    return sample_info;
}

static __host__ __device__ float getNextQValue(float3 &position, float3 &normal, float3 &wo, float3 &wi, float pdf)
{
    if(pdf <= 1e-5){
        return 0.0;
    }
    float new_value;
    // Expected SARSA
    if(q_table_update_method == 0){
        new_value = getExpectedQValue(position, normal, wo);
    }
    // Q-learning
//    else if(q_table_update_method == 1){
//        new_value = getMaximumQValue(position, normal, wo);
//    }
    // SARSA
    else if(q_table_update_method == 1){
        new_value = getQValue(position, wi) * max(0.0f, dot(wi, normal)) / (pdf * M_PIf);
    }
    // SARSA2
    else if(q_table_update_method == 2){
        new_value = getQValue(position, wi) * max(0.0f, dot(wi, normal)) * 2;
    }
    return new_value;
}

static __host__ __device__ float getNextQValueVolume(float g, float3 &position, float3 &wo, float3 &wi)
{
    float new_value;
    // Expected SARSA
    if(q_table_update_method == 0){
        new_value = getExpectedQValueVolume(g, position, wo);
    }
    // Q-learning
    else if(q_table_update_method == 1){
        new_value = getMaximumQValueVolume(g, position, wo);
    }
    // SARSA
    else if(q_table_update_method == 2){
        new_value = getQValue(position, wi);
    }
    return new_value;
}