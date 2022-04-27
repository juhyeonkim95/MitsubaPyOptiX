# Config Details

### scene_name
The name of the scene.

### scene_file_path
The file path to the scene.

### scale
Divide original image size by this value.

### max_depth
Maximum path tracing depth.

### rr_begin_depth
Russian Roulette begin depth.

### use_next_event_estimation
Use NEE or direct light sampling.

### spp
Samples per pixel (spp) budget.

### time_limit_in_sec
Time (sec) budget. If this is -1, spp budget is used.

### sampling_strategy
Sampling strategy. Should be one of following.
* brdf or bsdf : sample proportional to BSDF
* mis : use BSDF sampling and radiance sampling
* qcos_inversion : sample proportional to product of radiance and cosine value using inversion sampling.
* qcos_reject : same with above using rejection sampling.
* qcos_reject_mix : same with above using optimized rejection sampling.

### q_table_update_method
Radiance record or Q table update method. Should be one of following.
* mc (Monte Carlo)
* sarsa
* expected_sarsa

### directional_mapping_method
Directional mapping method. Should be one of following.
* cylindrical
* shirley

### spatial_data_structure_type
Spatial data structure for radiance record. Should be one of following.
* grid or voxel
* octree
* binary_tree

### directional_data_structure_type
Directional data structure for radiance record. Should be one of following.
* grid
* quad_tree

### clear_accumulated_info_per_update
Clear accumulated information such as radiance, visited number per update.
It can be only used when data structure is not changed.

### accumulative_q_table_update
If set true, update q table by accumulation.

### learning_method
Should be one of incremental of exponential.

### bsdf_sampling_fraction
BSDF sampling fraction for MIS sampling.

### n_cube
Used for the size of spatial data structure.

### n_uv
Used for the size of directional data structure.


### show_picture
If true, show picture


