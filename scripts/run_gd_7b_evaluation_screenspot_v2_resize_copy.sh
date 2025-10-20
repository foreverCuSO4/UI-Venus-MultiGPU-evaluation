#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=3,6,7

models=("ui_venus_ground_7b")
for model in "${models[@]}"
do
    accelerate launch --num_processes=3 models/grounding/eval_screenspot_pro_multiGPU.py  \
        --model_type ${model}  \
        --screenspot_imgs "../ScreenSpot-v2-variants/image_0.9"  \
        --screenspot_test "../ScreenSpot-v2-variants/annotations"  \
        --model_name_or_path "inclusionAI/UI-Venus-Ground-7B" \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "venus_7b_ssv2_0.9/venus_7b_ss2.json" \
        --inst_style "instruction" \
        --bbox "xywh" \
        --compression_ratio 0.9

done

models=("ui_venus_ground_7b")
for model in "${models[@]}"
do
    accelerate launch --num_processes=3 models/grounding/eval_screenspot_pro_multiGPU.py  \
        --model_type ${model}  \
        --screenspot_imgs "../ScreenSpot-v2-variants/image_0.8"  \
        --screenspot_test "../ScreenSpot-v2-variants/annotations"  \
        --model_name_or_path "inclusionAI/UI-Venus-Ground-7B" \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "venus_7b_ssv2_0.8/venus_7b_ss2.json" \
        --inst_style "instruction" \
        --bbox "xywh" \
        --compression_ratio 0.8

done

models=("ui_venus_ground_7b")
for model in "${models[@]}"
do
    accelerate launch --num_processes=3 models/grounding/eval_screenspot_pro_multiGPU.py  \
        --model_type ${model}  \
        --screenspot_imgs "../ScreenSpot-v2-variants/image_0.7"  \
        --screenspot_test "../ScreenSpot-v2-variants/annotations"  \
        --model_name_or_path "inclusionAI/UI-Venus-Ground-7B" \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "venus_7b_ssv2_0.7/venus_7b_ss2.json" \
        --inst_style "instruction" \
        --bbox "xywh" \
        --compression_ratio 0.7

done
