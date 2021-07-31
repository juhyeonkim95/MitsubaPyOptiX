#include <stddef.h>

float cmult(int int_param, float float_param) {
    float return_value = int_param * float_param;
    printf("    In cmult : int: %d float %.1f returning  %.1f\n", int_param,
            float_param, return_value);
    return return_value;
}
int value = 1;

void cfun(const double *indatav, size_t size, double *outdatav)
{
    size_t i;
    for (i = 0; i < size; ++i)
        outdatav[i] = indatav[i] * value;
}

//
//struct QuadTree
//{
//    unsigned int *dtree_index_array;
//    unsigned int *dtree_rank_array;
//    unsigned int *dtree_depth_array;
//    unsigned int *dtree_select_array;
//    float *dtree_value_array;
//    unsigned int *dtree_current_size_array;
//}
//
#include <math.h>

float get_total_irradiance(
    const int *dtree_index_array,
    const float *dtree_value_array,
    const int *dtree_depth_array,
    size_t current_size,
    size_t n_s,
    size_t index)
{
    float irradiance = 0;
    size_t offset = index * n_s;
    for(int i=0; i<current_size; i++){
        size_t index = dtree_index_array[offset + i];
        float value = dtree_value_array[offset + i];
        size_t depth = dtree_depth_array[offset + i];
        if(dtree_index_array[offset+i] == 0){
            irradiance += value * pow(0.25f, depth);
        }
    }
    return irradiance;
}

//
//
//void update_quadtree(
//    const int *index_array,
//
//)
//{
//     const float threshold = 0.01f;
//     float total_irradiance = get_irradiance();
//}