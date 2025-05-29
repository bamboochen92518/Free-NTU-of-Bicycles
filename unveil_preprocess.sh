#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 model_name gpu_id iteration"
    exit 1
fi

model_name=$1
gpu_id=$2
iteration=$3

# Run the Python scripts with the specified model, GPU, and iteration
CUDA_VISIBLE_DEVICES=$gpu_id python inpainting_pipeline/1_selection/1_instance_visualization.py -m $model_name --mask_vehicle --load_iteration $iteration # --mask_vehicle option can be modified to change the semantic of the removed objects

CUDA_VISIBLE_DEVICES=$gpu_id python inpainting_pipeline/2_condition_preparation/1_select_instance.py -m $model_name --all --load_iteration $iteration

CUDA_VISIBLE_DEVICES=$gpu_id python inpainting_pipeline/2_condition_preparation/2_generate_inpainted_mask.py -m $model_name --load_iteration $iteration