#include <stddef.h>
#include <math.h>
#include <string.h>

void build_rank_array(
    unsigned int **dtree_index_array,
    unsigned int **dtree_rank_array,
    unsigned int **dtree_depth_array,
    unsigned int **dtree_select_array,
    float **dtree_value_array,
    unsigned int *dtree_current_size_array,
    unsigned int launch_index
)
{
    unsigned int sum = 0;
    unsigned int current_size = dtree_current_size_array[launch_index];
        for(int i=0; i< current_size; i++){
        dtree_rank_array[launch_index][i] = sum;
        sum += dtree_index_array[launch_index][i];
    }

    for(int i=0; i< current_size; i++){
        if(dtree_index_array[launch_index][i] == 1){
            for(int j=0; j<4; j++){
                unsigned int child_j = 4 * dtree_rank_array[launch_index][i] + j + 1;
                dtree_select_array[launch_index][child_j] = i;
            }
        }
    }
}

void update_parent_radiance(
    unsigned int **dtree_index_array,
    unsigned int **dtree_rank_array,
    unsigned int **dtree_depth_array,
    unsigned int **dtree_select_array,
    float **dtree_value_array,
    unsigned int *dtree_current_size_array,
    unsigned int launch_index
){
    unsigned int current_size = dtree_current_size_array[launch_index];
    for(int i=current_size - 1; i>=1 ; i--){
        unsigned int parent_id = dtree_select_array[launch_index][i];
        dtree_value_array[launch_index][parent_id] +=  dtree_value_array[launch_index][i]* 0.25;
    }
}

void update_quadtree_single(
    unsigned int **dtree_index_array,
    unsigned int **dtree_rank_array,
    unsigned int **dtree_depth_array,
    unsigned int **dtree_select_array,
    float **dtree_value_array,
    unsigned int *dtree_current_size_array,
    unsigned int launch_index,
    float threshold
)
{
    unsigned int current_size_original = dtree_current_size_array[launch_index];

    update_parent_radiance(dtree_index_array, dtree_rank_array, dtree_depth_array, dtree_select_array, dtree_value_array, dtree_current_size_array, launch_index);
    float total_irradiance = dtree_value_array[launch_index][0];//get_total_irradiance(dtree_index_array, dtree_rank_array, dtree_depth_array, dtree_select_array, dtree_value_array, dtree_current_size_array, launch_index);


    const unsigned int MAX_QUEUE_SIZE = 2048;
    const unsigned int MAX_QUADTREE_SIZE = 512;

    unsigned int index_queue[MAX_QUEUE_SIZE];
    float value_queue[MAX_QUEUE_SIZE];
    unsigned int depth_queue[MAX_QUEUE_SIZE];

    unsigned int index_array[MAX_QUADTREE_SIZE];
    float value_array[MAX_QUADTREE_SIZE];
    unsigned int depth_array[MAX_QUADTREE_SIZE];

    unsigned int current_q_size = 0;
    unsigned int current_q_pointer_left = 0;
    unsigned int current_q_pointer_right = 0;
    unsigned int current_array_size = 0;

    index_queue[0] = 0;
    value_queue[0] = dtree_value_array[launch_index][0];
    depth_queue[0] = 0;
    current_q_pointer_right += 1;
    current_q_size += 1;

    unsigned int current_quadtree_size = 1;

    while(current_q_size > 0){
        unsigned int node = index_queue[current_q_pointer_left];
        float val = value_queue[current_q_pointer_left];
        unsigned int depth = depth_queue[current_q_pointer_left];
        float local_irradiance = val * powf(0.25f, depth);

        current_q_pointer_left += 1;
        current_q_size -= 1;

        depth_array[current_array_size] = depth;
        value_array[current_array_size] = val;

        if((local_irradiance > total_irradiance * threshold) && (current_quadtree_size + 4 <= MAX_QUADTREE_SIZE)){
            index_array[current_array_size] = 1;
            current_array_size += 1;

            // internal node
            if(node < current_size_original && dtree_index_array[launch_index][node] == 1){
                for(unsigned int i=0; i<4; i++){
                    unsigned int child_id =  4 * dtree_rank_array[launch_index][node] + i + 1;
                    index_queue[current_q_pointer_right] = child_id;
                    value_queue[current_q_pointer_right] = dtree_value_array[launch_index][child_id];
                    depth_queue[current_q_pointer_right] = depth + 1;
                    current_q_pointer_right += 1;
                    current_q_size += 1;
                }
            }
            // leaf node
            else{
                for(unsigned int i=0; i<4; i++){
                    index_queue[current_q_pointer_right] = current_size_original; // not valid node index
                    value_queue[current_q_pointer_right] = val;
                    depth_queue[current_q_pointer_right] = depth + 1;
                    current_q_pointer_right += 1;
                    current_q_size += 1;
                }
            }
            current_quadtree_size += 4;
        }
        else {
            index_array[current_array_size] = 0;
            current_array_size += 1;
        }
    }

    // assert current_array_size == current_quadtree_size;

    // replace old one
    memcpy(dtree_index_array[launch_index], index_array, current_array_size*sizeof(unsigned int));
    memcpy(dtree_value_array[launch_index], value_array, current_array_size*sizeof(float));
    memcpy(dtree_depth_array[launch_index], depth_array, current_array_size*sizeof(unsigned int));

//    for(int i=0; i<current_array_size; i++){
//        dtree_index_array[launch_index][i] = index_array[i];
//        dtree_value_array[launch_index][i] = value_array[i];
//        dtree_depth_array[launch_index][i] = depth_array[i];
//    }
    dtree_current_size_array[launch_index] = current_array_size;

    build_rank_array(dtree_index_array, dtree_rank_array, dtree_depth_array, dtree_select_array, dtree_value_array, dtree_current_size_array, launch_index);
    // update_parent_radiance(dtree_index_array, dtree_rank_array, dtree_depth_array, dtree_select_array, dtree_value_array, dtree_current_size_array, launch_index);
}

void update_quadtree(
    unsigned int **dtree_index_array,
    unsigned int **dtree_rank_array,
    unsigned int **dtree_depth_array,
    unsigned int **dtree_select_array,
    float **dtree_value_array,
    unsigned int *dtree_current_size_array,
    unsigned int n_s,
    float threshold
){
    for(int i=0; i<n_s; i++){
        update_quadtree_single(dtree_index_array, dtree_rank_array, dtree_depth_array, dtree_select_array,dtree_value_array,dtree_current_size_array,i, threshold);
    }
}

void update_quadtree_multi(
    unsigned int **dtree_index_array,
    unsigned int **dtree_rank_array,
    unsigned int **dtree_depth_array,
    unsigned int **dtree_select_array,
    float **dtree_value_array,
    unsigned int *dtree_current_size_array,
    unsigned int n_s,
    float threshold
){
#pragma omp parallel for
    for(int i=0; i<n_s; i++){
        update_quadtree_single(dtree_index_array, dtree_rank_array, dtree_depth_array, dtree_select_array,dtree_value_array,dtree_current_size_array,i,threshold);
    }
}

unsigned int update_binary_tree_native(
    unsigned int **dtree_index_array,
    unsigned int **dtree_rank_array,
    unsigned int **dtree_depth_array,
    unsigned int **dtree_select_array,
    float **dtree_value_array,
    unsigned int *dtree_current_size_array,

    unsigned int *visit_count_array,
    unsigned int *child_array,
    unsigned int *parent_array,
    unsigned int *axis_array,
    unsigned int *leaf_index_array,

    unsigned int threshold,
    unsigned int leaf_node_number,
    unsigned int max_leaf_node_number
)
{
    const int MAX_QUEUE_SIZE = 512 * 32;
    unsigned int node_queue[MAX_QUEUE_SIZE];
    unsigned int current_q_pointer_right = 0;
    unsigned int current_q_pointer_left = 0;
    unsigned int current_q_size =0;
    unsigned int total_node_number = 2  * leaf_node_number - 1;


    node_queue[0] = 0;
    current_q_pointer_right += 1;
    current_q_size += 1;

    while(current_q_size > 0){
        if (leaf_node_number >= max_leaf_node_number)
            break;

        unsigned int node = node_queue[current_q_pointer_left];
        current_q_pointer_left += 1;
        current_q_size -= 1;

        // leaf node
        if(child_array[node] == 0){
            unsigned int node_leaf_index = leaf_index_array[node];
            unsigned int visited_count = visit_count_array[node_leaf_index];
            if(visited_count > threshold){
                if(leaf_node_number + 2 > max_leaf_node_number)
                    break;
                child_array[node] = total_node_number;

                for(int i=0; i<2; i++){
                    unsigned int idx = total_node_number + i;
                    axis_array[idx] = (axis_array[node] + 1) % 3;
                    parent_array[idx] = node;
                }

                // First child node : only copy leaf node index except visit count which should be halved
                unsigned int child_1_leaf_index = node_leaf_index;
                leaf_index_array[total_node_number] = child_1_leaf_index;
                visit_count_array[child_1_leaf_index] = visited_count / 2;

                // Second child node : copy data
                unsigned int child_2_leaf_index = leaf_node_number;
                leaf_index_array[total_node_number + 1] = child_2_leaf_index;
                visit_count_array[child_2_leaf_index] = visited_count / 2;

                // data copy
                unsigned int dtree_size = dtree_current_size_array[node_leaf_index];
                dtree_current_size_array[child_2_leaf_index] = dtree_size;

                memcpy(dtree_index_array[child_2_leaf_index], dtree_index_array[node_leaf_index], dtree_size*sizeof(unsigned int));
                memcpy(dtree_rank_array[child_2_leaf_index], dtree_rank_array[node_leaf_index], dtree_size*sizeof(unsigned int));
                memcpy(dtree_depth_array[child_2_leaf_index], dtree_depth_array[node_leaf_index], dtree_size*sizeof(unsigned int));
                memcpy(dtree_select_array[child_2_leaf_index], dtree_select_array[node_leaf_index], dtree_size*sizeof(unsigned int));
                memcpy(dtree_value_array[child_2_leaf_index], dtree_value_array[node_leaf_index], dtree_size*sizeof(float));

                leaf_node_number += 1;
                total_node_number += 2;
            }
        }

        // inner node
        if(child_array[node] != 0) {
            unsigned int child_idx = child_array[node];
            node_queue[current_q_pointer_right] = child_idx;
            node_queue[current_q_pointer_right+1] = child_idx+1;
            current_q_size += 2;
            current_q_pointer_right += 2;
        }
    }

    return leaf_node_number;
}

unsigned int update_binary_tree_native_grid(
    float **dtree_value_array,
    unsigned int dtree_size,

    unsigned int *visit_count_array,
    unsigned int *child_array,
    unsigned int *parent_array,
    unsigned int *axis_array,
    unsigned int *leaf_index_array,

    unsigned int threshold,
    unsigned int leaf_node_number,
    unsigned int max_leaf_node_number
)
{
    const int MAX_QUEUE_SIZE = 512 * 32;
    unsigned int node_queue[MAX_QUEUE_SIZE];
    unsigned int current_q_pointer_right = 0;
    unsigned int current_q_pointer_left = 0;
    unsigned int current_q_size =0;
    unsigned int total_node_number = 2  * leaf_node_number - 1;


    node_queue[0] = 0;
    current_q_pointer_right += 1;
    current_q_size += 1;

    while(current_q_size > 0){
        if (leaf_node_number >= max_leaf_node_number)
            break;

        unsigned int node = node_queue[current_q_pointer_left];
        current_q_pointer_left += 1;
        current_q_size -= 1;

        // leaf node
        if(child_array[node] == 0){
            unsigned int node_leaf_index = leaf_index_array[node];
            unsigned int visited_count = visit_count_array[node_leaf_index];
            if(visited_count > threshold){
                if(leaf_node_number + 2 > max_leaf_node_number)
                    break;
                child_array[node] = total_node_number;

                for(int i=0; i<2; i++){
                    unsigned int idx = total_node_number + i;
                    axis_array[idx] = (axis_array[node] + 1) % 3;
                    parent_array[idx] = node;
                }

                // First child node : only copy leaf node index except visit count which should be halved
                unsigned int child_1_leaf_index = node_leaf_index;
                leaf_index_array[total_node_number] = child_1_leaf_index;
                visit_count_array[child_1_leaf_index] = visited_count / 2;

                // Second child node : copy data
                unsigned int child_2_leaf_index = leaf_node_number;
                leaf_index_array[total_node_number + 1] = child_2_leaf_index;
                visit_count_array[child_2_leaf_index] = visited_count / 2;

                // data copy
                memcpy(dtree_value_array[child_2_leaf_index], dtree_value_array[node_leaf_index], dtree_size*sizeof(float));

                leaf_node_number += 1;
                total_node_number += 2;
            }
        }

        // inner node
        if(child_array[node] != 0) {
            unsigned int child_idx = child_array[node];
            node_queue[current_q_pointer_right] = child_idx;
            node_queue[current_q_pointer_right+1] = child_idx+1;
            current_q_size += 2;
            current_q_pointer_right += 2;
        }
    }

    return leaf_node_number;
}