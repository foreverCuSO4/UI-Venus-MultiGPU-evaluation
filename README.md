# UI-Venus (Multi GPU Accelerated Evaluation Fork)

This is a high-performance fork of the original [inclusionAI/UI-Venus](https://github.com/inclusionAI/UI-Venus) project, optimized for rapid, robust, multi-GPU evaluation.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Original Report](https://img.shields.io/badge/Technical%20Report-arXiv-blueviolet)](http://arxiv.org/abs/2508.10833)
[![Original Project](https://img.shields.io/badge/Original%20Project-UI--Venus-green?logo=github)](https://github.com/inclusionAI/UI-Venus)

## ✨ Fork Enhancements

This fork supercharges the original UI-Venus evaluation pipeline using Hugging Face `accelerate`.

*   🚀 **Multi-GPU Acceleration**: Dramatically reduces evaluation time by running inference in parallel across all available GPUs.
*   📊 **Real-time Feedback**: Monitor model performance live. Each GPU prints its evaluation results and running accuracy in real-time.
*   💾 **Crash-Resilient Saving**: Live-saves results from each GPU to a separate file, ensuring no progress is lost during long evaluation runs. Results are automatically merged upon completion.

## 🔧 Installation

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt accelerate
    ```

2.  **Configure Accelerate**:
    Run the interactive configuration wizard once. For most cases, the default options are sufficient.
    ```bash
    accelerate config
    ```
    *(For best performance, we recommend using `torch.compile` with the `inductor` backend and `bf16` precision when prompted.)*

## 🚀 Quick Start: Multi-GPU Evaluation

### One-Click Evaluation

Use the provided shell scripts to launch the evaluation on a specific dataset.

-   **For ScreenSpot-Pro (7B Model):**
    ```bash
    bash scripts/run_gd_7b_evaluation_screenspot_pro.sh
    ```

-   **For ScreenSpot-v2 (7B Model):**
    ```bash
    bash scripts/run_gd_7b_evaluation_screenspot_v2.sh
    ```

🔧 **Configuration Required**: Before running, you may need to **edit these scripts** to set the correct paths for your datasets and model checkpoints.

### Running on Specific GPUs

To control which GPUs are used for the evaluation, **edit the launch scripts** inside the `scripts/` directory (e.g., `run_gd_7b_evaluation_screenspot_pro.sh`). You can use one of the following methods inside the script.

**Method 1: Using `CUDA_VISIBLE_DEVICES` (Recommended)**
Set the environment variable at the top of the script to control GPU visibility. The `--num_processes` flag should match the number of GPUs.

```bash
# Inside run_gd_7b_evaluation_screenspot_pro.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

accelerate launch --num_processes=$NUM_GPUS models/grounding/eval_screenspot_pro.py [YOUR_ARGS...]
```

**Method 2: Using the `--gpu_ids` flag**
Alternatively, add the `--gpu_ids` flag directly to the `accelerate launch` command.

```bash
# Inside run_gd_7b_evaluation_screenspot_pro.sh
accelerate launch --gpu_ids '0,1,2,3' models/grounding/eval_screenspot_pro.py [YOUR_ARGS...]
```

## Original Project Information

This fork is built upon the excellent work of the UI-Venus team. For details on the model architecture, training data, and original research contributions, please refer to the **[original repository](https://github.com/inclusionAI/UI-Venus)**.

All performance benchmarks are from the original project and credited to its authors.

## ©️ Citation

If you use this work, please cite the original paper:
```plain
@misc{gu2025uivenustechnicalreportbuilding,
      title={UI-Venus Technical Report: Building High-performance UI Agents with RFT}, 
      author={...},
      year={2025},
      eprint={2508.10833},
      ...
}
```