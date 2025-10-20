#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

model="ui_venus_ground_7b"
model_name_or_path="inclusionAI/UI-Venus-Ground-7B"
screenspot_test="../ScreenSpot-v2-variants/annotations"
task="all"
language="en"
gt_type="positive"
inst_style="instruction"
bbox="xywh"

# 使用整数表示压缩率（9 表示 0.9，8 表示 0.8，...，1 表示 0.1）
for cr_int in {9..1}; do
    cr_rounded="0.0${cr_int}"  # 生成 "0.9", "0.8", ..., "0.1"

    screenspot_imgs="../ScreenSpot-v2-variants/image_${cr_rounded}"
    log_dir="venus_7b_ssv2_${cr_rounded}"
    log_path="${log_dir}/venus_7b_ss2.json"

    mkdir -p "$log_dir"

    echo "=================================================="
    echo "Running evaluation for compression ratio: ${cr_rounded}"
    echo "Images dir: ${screenspot_imgs}"
    echo "Log path: ${log_path}"
    echo "=================================================="

    accelerate launch --num_processes=1 models/grounding/eval_screenspot_pro_multiGPU.py \
        --model_type "${model}" \
        --screenspot_imgs "${screenspot_imgs}" \
        --screenspot_test "${screenspot_test}" \
        --model_name_or_path "${model_name_or_path}" \
        --task "${task}" \
        --language "${language}" \
        --gt_type "${gt_type}" \
        --log_path "${log_path}" \
        --inst_style "${inst_style}" \
        --bbox "${bbox}" \
        --compression_ratio "${cr_rounded}"
done

echo "✅ All evaluations completed."