#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1

models=("ui_venus_ground_7b") 
for model in "${models[@]}"
do
    accelerate launch --num_processes=2 models/grounding/eval_screenspot_pro_multiGPU.py  \
        --model_type ${model}  \
        --screenspot_imgs "Screenspot-pro/images"  \
        --screenspot_test "Screenspot-pro/annotations"  \
        --model_name_or_path "inclusionAI/UI-Venus-Ground-7B" \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "venus_7b/venus_7b_pro.json" \
        --inst_style "instruction" \
        --bbox "xyxy"

done