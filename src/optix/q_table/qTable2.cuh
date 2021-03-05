#include <optixu/optixu_math_namespace.h>
#include "optix/common/helpers.h"
#include "optix/common/prd_struct.h"
#include "optix/common/random.h"
#include "optix/common/sampling.h"

// #include "optix/bsdf/bsdf.h"


using namespace optix;

rtBuffer<float, 2>              q_table;
rtBuffer<float, 2>              irradiance_table;
rtBuffer<float, 2>              max_radiance_table;

rtBuffer<float, 2>              q_table_accumulated;
rtBuffer<float, 2>              q_table_old;
rtBuffer<uint, 2>               visit_counts;
rtBuffer<float2, 2>              mcmc_table;



rtDeclareVariable(uint3,        unitCubeNumber, , );
rtDeclareVariable(uint2,        unitUVNumber, , );
rtDeclareVariable(float3,       unitCubeSize, , );

rtDeclareVariable(unsigned int,     q_table_update_method, , );
rtDeclareVariable(unsigned int,     q_value_sample_method, , );
rtDeclareVariable(unsigned int,     save_q_cos, , );

rtDeclareVariable(float,     q_value_sample_constant, , );

static __host__ __device__ bool checkBoundary(float3 position)
{
    bool valid_x = position.x > 0 && position.x < unitCubeNumber.x * unitCubeSize.x;
    bool valid_y = position.y > 0 && position.y < unitCubeNumber.y * unitCubeSize.y;
    bool valid_z = position.z > 0 && position.z < unitCubeNumber.z * unitCubeSize.z;
    return valid_x && valid_y && valid_z;
}

static __host__ __device__ uint positionToIndex(float3 position)
{
    float3 abs_pos = make_float3(abs(position.x), abs(position.y), abs(position.z));
    uint3 idx = make_uint3(abs_pos / unitCubeSize);
    idx = clamp(idx, make_uint3(0), unitCubeNumber - make_uint3(1));
    uint idx_int = (idx.z * unitCubeNumber.y * unitCubeNumber.z) + (idx.y * unitCubeNumber.z) + idx.x;
    return idx_int;
}

static __host__ __device__ uint directionToIndex(float3 direction)
{
    direction = normalize(direction);
    float2 uv = mapDirectionToUV(direction);

    uint u = uint(uv.x * float(unitUVNumber.x));
    uint v = uint(uv.y * float(unitUVNumber.y));

    u = clamp(u, uint(0), unitUVNumber.x - 1);
    v = clamp(v, uint(0), unitUVNumber.y - 1);

    if(direction.y < 0){
        u += unitUVNumber.x;
    }
    uint idx = u * unitUVNumber.y + v;

    return idx;//clamp(uint(direction.x * 16), uint(0), uint(16));
}

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
    uint rayIndex = directionToIndex(direction);
    return q_table_old[make_uint2(positionIndex, rayIndex)];
}

static __host__ __device__ void setQValue(float3 position, float3 direction, float value)
{
    uint positionIndex = positionToIndex(position);
    uint rayIndex = directionToIndex(direction);
    q_table[make_uint2(positionIndex, rayIndex)] = value;
}

static __host__ __device__ void accumulateQValue(float3 position, float3 direction, float value)
{
    uint positionIndex = positionToIndex(position);
    uint rayIndex = directionToIndex(direction);
    uint2 idx = make_uint2(positionIndex, rayIndex);
    atomicAdd(&q_table_accumulated[idx], value);
    atomicAdd(&visit_counts[idx], 1);
}


static __host__ __device__ uint updateVisit(float3 position, float3 direction)
{
    uint positionIndex = positionToIndex(position);
    uint rayIndex = directionToIndex(direction);
    uint2 idx = make_uint2(positionIndex, rayIndex);
    float v = visit_counts[idx];
    atomicAdd(&visit_counts[idx], 1);
    //visit_counts[idx] = v + 1;

    return v;
}

static __host__ __device__ uint getVisit(float3 position, float3 direction)
{
    uint positionIndex = positionToIndex(position);
    uint rayIndex = directionToIndex(direction);
    uint2 idx = make_uint2(positionIndex, rayIndex);
    float v = visit_counts[idx];
    return v;
}

static __host__ __device__ void setQValueSoft(float3 position, float3 target_position, float target_q_value)
{
    for(int dx=-1; dx<=1; dx++){
        for(int dy=-1; dy<=1; dy++){
            for(int dz=-1; dz<=1; dz++){
                if(dx == 0 && dy == 0 && dz == 0)
                    continue;
                float3 r = unitCubeSize;
                float3 ray_origin = position + make_float3(dx, dy, dz) * r;
                if(!checkBoundary(ray_origin))
                    continue;
                float3 ray_direction = target_position - ray_origin;
                ray_direction = ray_direction/length(ray_direction);

                float alpha = 1.0f / sqrt(1.0f + getVisit(ray_origin, ray_direction));
                //alpha *= 0.5;
                float update_value = (1-alpha) * getQValue(ray_origin, ray_direction) + alpha * target_q_value;
                setQValue(ray_origin, ray_direction, update_value);
            }
        }
    }
}

static __host__ __device__ float getPolicyQValue(unsigned int positionIndex, unsigned int i){
//    float target_value = 0;
//    float bias_p = 0.01f;
//    if(q_value_sample_method == 1){
//        target_value = q_table_old[make_uint2(positionIndex, i)] + bias_p;
//    } else if (q_value_sample_method == 2){
//        target_value = q_table_old[make_uint2(positionIndex, i)] + bias_p;
//        target_value = pow(target_value, q_value_sample_constant);
//        //target_value * target_value;
//    }
    return q_table_old[make_uint2(positionIndex, i)] + 0.01f;
}

//static __host__ __device__ sampleScatteringDirectionProportionalToQ()
//{
//    uint positionIndex = positionToIndex(position);
//}

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
    uint n_i = directionToIndex(normal);
    float q_value_sum = 0;
    if(irradiance_table[make_uint2(positionIndex, n_i)] > 0.0){
        q_value_sum = irradiance_table[make_uint2(positionIndex, n_i)];
    } else {
        for(unsigned int i=0; i<UVN * 2; i++){
            float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
            float cos_theta = dot(normal, v_i);
            if(cos_theta > 0.0f){
                float c = cos_theta;//considerCosineTerm?cos_theta:1.0;
                q_value_sum += getPolicyQValue(positionIndex, i) * c;
            }
        }
        irradiance_table[make_uint2(positionIndex, n_i)] = q_value_sum;
    }

    Sample_info sample_info;

    float2 x1 = mcmc_table[make_uint2(positionIndex, n_i)];
    float2 x2 = mutate(x1, seed);

    optix::Onb onb( normal );
    float3 v1 = UniformSampleHemisphere(x1.x, x1.y);
    onb.inverse_transform(v1);
    float3 v2 = UniformSampleHemisphere(x2.x, x2.y);
    onb.inverse_transform(v2);
    //float3 v1 = mapThetaPhiToDirection(x1);
    //float3 v2 = mapThetaPhiToDirection(x2);

    uint uv1 = directionToIndex(v1);
    uint uv2 = directionToIndex(v2);

    float q1 = getPolicyQValue(positionIndex, uv1) * max(0.0, dot(normal, v1));
    float q2 = getPolicyQValue(positionIndex, uv2) * max(0.0, dot(normal, v2));
    float pdf;
    q1 += 1e-7;
    q2 += 1e-7;

    float a = min(1.0, q2 / q1);
    float3 direction;
    if(rnd(seed) < a){
        x1 = x2;
        mcmc_table[make_uint2(positionIndex, n_i)] = x2;
        direction = v2;
        pdf = q2;
    } else{
        direction = getDirectionFrom(uv1, make_float2(rnd(seed), rnd(seed)));
        pdf = q1;
    }
    pdf /= q_value_sum;
    pdf *= float(UVN);
    pdf *= 1/(2*M_PIf);
    pdf = max(pdf, 0.01f);

    sample_info.direction = direction;
    sample_info.pdf = pdf;
    return sample_info;
}

static __host__ __device__ Sample_info sampleScatteringDirectionProportionalToQReject(float3 position, float3 normal, unsigned int &seed)
{
    uint UVN = unitUVNumber.x * unitUVNumber.y;
    uint positionIndex = positionToIndex(position);
    uint n_i = directionToIndex(normal);
    optix::Onb onb( normal );
    float q_value_sum = 0;
    float max_val = 0;
    if(irradiance_table[make_uint2(positionIndex, n_i)] > 0.0){
        q_value_sum = irradiance_table[make_uint2(positionIndex, n_i)];
        max_val = max_radiance_table[make_uint2(positionIndex, n_i)];
    } else {

        for(unsigned int i=0; i<UVN * 2; i++){
            float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
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
        irradiance_table[make_uint2(positionIndex, n_i)] = q_value_sum;
        max_radiance_table[make_uint2(positionIndex, n_i)] = max_val;
    }

    Sample_info sample_info;
    float3 direction;
    float pdf;

    while(true){
        float2 x = make_float2(rnd(seed), rnd(seed));
        direction = UniformSampleHemisphere(x.x, x.y);
        onb.inverse_transform(direction);
        float e = rnd(seed);
        uint uv = directionToIndex(direction);
        pdf = getPolicyQValue(positionIndex, uv) * max(0.0, dot(normal, direction));
        if(e < pdf / max_val){
            break;
        }
    }

    pdf /= q_value_sum;
    pdf *= float(UVN);
    pdf *= 1/(2*M_PIf);
    pdf = max(pdf, 0.01f);

    sample_info.direction = direction;
    sample_info.pdf = pdf;
    return sample_info;
}

static __host__ __device__ Sample_info sampleScatteringDirectionProportionalToQMCMC2(float3 position, float3 normal, unsigned int &seed)
{
    uint UVN = unitUVNumber.x * unitUVNumber.y;
    uint n_i = directionToIndex(normal);
    uint positionIndex = positionToIndex(position);

    float2 x1 = mcmc_table[make_uint2(positionIndex, n_i)];
    float2 x2 = mutate(x1, seed);

    float3 v1 = mapThetaPhiToDirection(x1);
    float3 v2 = mapThetaPhiToDirection(x2);

    uint uv1 = directionToIndex(v1);
    uint uv2 = directionToIndex(v2);

    float q1 = getPolicyQValue(positionIndex, uv1);// * max(0.0, dot(normal, v1));
    float q2 = getPolicyQValue(positionIndex, uv2);// * max(0.0, dot(normal, v2));
    float pdf;
    q1 += 1e-7;
    q2 += 1e-7;

    float a = min(1.0, q2 / q1);
    float3 direction;
    if(rnd(seed) < a){
        x1 = x2;
        mcmc_table[make_uint2(positionIndex, n_i)] = x2;
        direction = v2;
        pdf = q2;
    } else{
        direction = getDirectionFrom(uv1, make_float2(rnd(seed), rnd(seed)));
        pdf = q1;
    }
    pdf *= float(UVN);
    pdf *= 1/(4*M_PIf);
    pdf = max(pdf, 0.01f);

    Sample_info sample_info;
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
        //float3 v_i = unitUVVectors[i];
        float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
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
        //float3 v_i = unitUVVectors[i];
        float3 v_i = getDirectionFrom(i, make_float2(0.5, 0.5));
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
    pdf = max(pdf, 0.01f);
    sample_info.direction = random_direction;
    sample_info.pdf = pdf;
    return sample_info;
}


static __host__ __device__ Sample_info sampleScatteringDirectionProportionalToQSphere(float3 position, unsigned int &seed)
{
    uint positionIndex = positionToIndex(position);
    uint UVN = unitUVNumber.x * unitUVNumber.y;
    Sample_info sample_info;
    float q_value_sum = 0;
    for(unsigned int i=0; i<UVN * 2; i++){
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

    float3 random_direction = getDirectionFrom(index, make_float2(rnd(seed), rnd(seed)));
    float pdf = (getPolicyQValue(positionIndex, index)) / q_value_sum;
    pdf *= float(UVN);
    pdf *= 1/(4*M_PIf);
    pdf = max(pdf, 0.01f);
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
            q_value_sum += getPolicyQValue(positionIndex, i);//#q_table_old[make_uint2(positionIndex, i)];
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
        float3 wo = getDirectionFrom(i, make_float2(0.5, 0.5));
        //float3 wo_local = to_local(onb, wo);
        //float3 fcos = Eval(mat, normal, wi_local, wo_local);
        float q_next = q_table[make_uint2(positionIndex, i)];
        float cos_theta = max(0.0, dot(normal, wo));
        q_value_sum += q_next * cos_theta;
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

        float q_next = q_table[make_uint2(positionIndex, i)];
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
        float v = q_table[make_uint2(positionIndex, i)];
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
        float v = q_table[make_uint2(positionIndex, i)];
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

static __host__ __device__ float getNextQValue(float3 &position, float3 &normal, float3 &wo, float3 &wi)
{
    float new_value;
    // Expected SARSA
    if(q_table_update_method == 0){
        new_value = getExpectedQValue(position, normal, wo);
    }
    // Q-learning
    else if(q_table_update_method == 1){
        new_value = getMaximumQValue(position, normal, wo);
    }
    // SARSA
    else if(q_table_update_method == 2){
        new_value = getQValue(position, wi) * 2 * max(0.0f, dot(wi, normal));
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