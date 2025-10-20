import copy
import itertools

import torch
import json
import re
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm

from accelerate import Accelerator

#logging.basicConfig(level=logging.INFO)


logging.basicConfig(level=logging.INFO)
#torch.manual_seed(114514)

GT_TYPES = ['positive', 'negative']
INSTRUCTION_STYLES = ['instruction', 'action', 'description']
LANGUAGES = ['en', 'cn']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=False)
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--inst_style', type=str, required=True, choices=INSTRUCTION_STYLES + ['all'], help="Instruction style to use.")
    parser.add_argument('--language', type=str, required=True, choices=LANGUAGES + ['all'], default='en', help="Language to use.")
    parser.add_argument('--gt_type', type=str, required=True, choices=GT_TYPES + ['all'], help="Ground truth type: 'positive' or 'negative'.")
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--bbox', type=str, required=True,default="xywh", help="bbox format in the dataset, xyxy or xywh", choices=['xyxy', 'xywh'])
    parser.add_argument('--compression_ratio', type=float, default=1.0, help="Image compression ratio to use, default 1.0 (no compression).")

    args = parser.parse_args()
    return args

def build_model(args):
    model_type = args.model_type
    model_name_or_path = args.model_name_or_path


    if model_type == "ui_venus_ground_7b":
        from ui_venus_ground_7b import UI_Venus_Ground_7B
        model = UI_Venus_Ground_7B()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()

    elif model_type == "ui_venus_ground_72b":
        from ui_venus_ground_72b import UI_Venus_Ground_72B
        model = UI_Venus_Ground_72B()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    
    else:
        raise ValueError(f"Unsupported model type {model_type}.")
    model.set_generation_config(temperature=0, max_new_tokens=256)
    return model

def collect_results_to_eval(results, platform=None, group=None, application=None, language=None, gt_type=None, instruction_style=None, ui_type=None):
    """
    Filters the results based on provided values. None means include all (ignore filtering this attribute).

    Parameters:
        results (list): A list of dictionaries containing sample results.
    
    Returns:
        list: A filtered list of dictionaries based on the given criteria.
    """
    filtered_results = []

    for sample in results:
        # Check each filter condition; if None, consider it as passed
        if (platform is None or sample.get("platform") == platform) and \
           (group is None or sample.get("group") == group) and \
           (application is None or sample.get("application") == application) and \
           (language is None or sample.get("language") == language) and \
           (gt_type is None or sample.get("gt_type") == gt_type) and \
           (instruction_style is None or sample.get("instruction_style") == instruction_style) and \
           (ui_type is None or sample.get("ui_type") == ui_type):
            filtered_results.append(sample)

    return filtered_results


def make_combinations(results, platform=False, group=None, application=False, language=False, gt_type=False, instruction_style=False, ui_type=False):
    """
    Returns a list of combinations of values for attributes where the corresponding parameter is set to True.
    """
    # Initialize a dictionary to store unique values for each attribute
    unique_values = {
        "platform": set(),
        "group": set(),
        "application": set(),
        "language": set(),
        "gt_type": set(),
        "instruction_style": set(),
        "ui_type": set(),
    }

    # Collect unique values from the results
    for sample in results:
        if platform:
            unique_values["platform"].add(sample.get("platform"))
        if group:
            unique_values["group"].add(sample.get("group"))
        if application:
            unique_values["application"].add(sample.get("application"))
        if language:
            unique_values["language"].add(sample.get("language"))
        if gt_type:
            unique_values["gt_type"].add(sample.get("gt_type"))
        if instruction_style:
            unique_values["instruction_style"].add(sample.get("instruction_style"))
        if ui_type:
            unique_values["ui_type"].add(sample.get("ui_type"))

    # Filter out the attributes that are set to False (no need for combinations)
    filtered_values = {key: list(value) for key, value in unique_values.items() if value}
    if not filtered_values:
        return []

    # Generate all combinations of the selected attributes using itertools.product
    attribute_combinations = list(itertools.product(*filtered_values.values()))

    # Convert combinations into dictionaries with corresponding attribute names
    combinations = []
    for combination in attribute_combinations:
        combinations.append(dict(zip(filtered_values.keys(), combination)))

    return combinations


def calc_metric_for_result_list(results):
    """Calculates the metrics for a simple result list."""
    num_total = len(results)
    correct_num = sum(1 for res in results if res["correctness"] == "correct")
    wrong_format_num = sum(1 for res in results if res["correctness"] == "wrong_format")

    # Calculate text and icon specific metrics using collect_results_to_eval
    text_results = collect_results_to_eval(results, ui_type="text")
    icon_results = collect_results_to_eval(results, ui_type="icon")

    text_correct = sum(1 for res in text_results if res["correctness"] == "correct")
    text_total = len(text_results)
    icon_correct = sum(1 for res in icon_results if res["correctness"] == "correct")
    icon_total = len(icon_results)
    metrics = {
        "num_correct_action": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / num_total if num_total > 0 else 0,
        "text_acc": text_correct / text_total if text_total > 0 else 0,
        "icon_acc": icon_correct / icon_total if icon_total > 0 else 0
    }
    return metrics


def eval_sample_positive_gt(sample, response, bbox_format, compressed_img_size, compression_ratio):
    bbox = sample["bbox"]
    if bbox_format == "xyxy":
        bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]  # x1, y1, x2, y2
    else:  # "xywh"
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # x1, y1, w, h
    original_img_size = sample["img_size"]  # 原始图像尺寸 (W_orig, H_orig)
    
    # 2. 计算缩放因子（基于面积压缩率）
    scale = compression_ratio ** 0.5

    # 3. 将原始 bbox 缩放到压缩后图像的坐标系
    if bbox_format == "xyxy":
        bbox_compressed = [
            bbox[0] * scale,
            bbox[1] * scale,
            bbox[2] * scale,
            bbox[3] * scale
        ]
    else:  # "xywh"
        bbox_compressed = [
            bbox[0] * scale,
            bbox[1] * scale,
            bbox[2] * scale,  # width
            bbox[3] * scale   # height
        ]
        # 转为 xyxy 用于判断
        x1 = bbox_compressed[0]
        y1 = bbox_compressed[1]
        x2 = x1 + bbox_compressed[2]
        y2 = y1 + bbox_compressed[3]
        bbox_compressed = [x1, y1, x2, y2]

    # 4. 获取预测点（归一化坐标）
    click_point = response["point"]  # [x_norm, y_norm] in [0,1]
    if click_point is None:
        return "wrong_format"

    # 5. 将预测点转为压缩图的像素坐标
    pred_x = click_point[0] * compressed_img_size[0]
    pred_y = click_point[1] * compressed_img_size[1]

    # 6. 判断是否在压缩图 bbox 内
    x1, y1, x2, y2 = bbox_compressed
    if x1 <= pred_x <= x2 and y1 <= pred_y <= y2:
        return "correct"
    else:
        return "wrong"
    
def eval_sample_negative_gt(sample, response):
    if response["result"] == "negative":
        return "correct"
    elif response["result"] == "positive":
        return "wrong"
    else: ## response["result"] == wrong_format
        return "wrong_format"

def evaluate_fine_grained(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        platform=True, 
        application=True,
        instruction_style=True, 
        gt_type=True
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        platform = combo.get("platform")
        application = combo.get("application")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            platform=platform,
            application=application,
            instruction_style=inst_style,
            gt_type=gt_type
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"plat:{platform} app:{application} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_seeclick_paper_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        platform=True, 
        instruction_style=True, 
        gt_type=True
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        platform = combo.get("platform")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            platform=platform,
            instruction_style=inst_style,
            gt_type=gt_type
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"plat:{platform} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_leaderboard_detailed_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        application=True,
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        application = combo.get("application")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            application=application,
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"app:{application}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_leaderboard_simple_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        group=True,
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        group = combo.get("group")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            group=group,
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"group:{group}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_overall(results):
    """
    Evaluates the overall metrics for all results without any filtering.
    
    Parameters:
        results (list): A list of dictionaries containing sample results.
        
    Returns:
        dict: A dictionary containing the overall metrics.
    """
    # Calculate metrics for the entire result set
    metrics = calc_metric_for_result_list(results)
    
    return metrics


def evaluate(results):
    """Collect results and calculate metrics. You can comment out function calls or add new ones based on your need.
    """
    result_report = {
        "details": [],  # Store detailed information for each sample
        "metrics": {}
    }

    # TODO: comment out function calls based on your need
    result_report["metrics"]["fine_grained"] = evaluate_fine_grained(results)
    result_report["metrics"]["seeclick_style"] = evaluate_seeclick_paper_style(results)
    result_report["metrics"]["leaderboard_simple_style"] = evaluate_leaderboard_simple_style(results)
    result_report["metrics"]["leaderboard_detailed_style"] = evaluate_leaderboard_detailed_style(results)
    result_report["metrics"]["overall"] = evaluate_overall(results)

    # Save detailed results
    result_report["details"] = results

    return result_report


def main():
    # initialize accelerator
    accelerator = Accelerator()
    torch.manual_seed(114514)
    args = parse_args()

    # build model
    model = build_model(args)
    model.model = accelerator.prepare(model.model)

    # Get the log path and prepare partial log path for each process
    log_path = args.log_path
    log_dir = os.path.dirname(log_path)
    log_filename = os.path.basename(log_path)
    partial_log_path = os.path.join(
        log_dir,
        f"{os.path.splitext(log_filename)[0]}_part_{accelerator.process_index}.json"
    )

    if accelerator.is_main_process:
        print("Preparing data... Each process will save its results live to a partial file.")
        print(f"Partial files will look like: {os.path.splitext(log_filename)[0]}_part_X.json")
        os.makedirs(log_dir, exist_ok=True)

    # Load and prepare data
    with accelerator.main_process_first():
        if args.task == "all":
            task_filenames = [
                os.path.splitext(f)[0]
                for f in os.listdir(args.screenspot_test)
                if f.endswith(".json")
            ]
        else:
            task_filenames = args.task.split(",")

        if args.inst_style == "all":
            inst_styles = INSTRUCTION_STYLES
        else:
            inst_styles = args.inst_style.split(",")

        if args.language == "all":
            languages = LANGUAGES
        else:
            languages = args.language.split(",")

        if args.gt_type == "all":
            gt_types = GT_TYPES
        else:
            gt_types = args.gt_type.split(",")

        tasks_to_run = []
        for task_filename in task_filenames:
            dataset = task_filename + ".json"
            with open(os.path.join(args.screenspot_test, dataset), 'r') as f:
                task_data = json.load(f)

            for inst_style in inst_styles:
                for gt_type in gt_types:
                    for lang in languages:
                        for task_instance in task_data:
                            task_instance = copy.deepcopy(task_instance)
                            task_instance["task_filename"] = task_filename
                            task_instance["gt_type"] = gt_type
                            task_instance["instruction_style"] = inst_style
                            task_instance["language"] = lang
                            if lang == "cn":
                                if inst_style != 'instruction' or gt_type != 'positive':
                                    raise AttributeError("Only positive samples and 'instruction' style are supported for Chinese instructions.")
                                task_instance["prompt_to_evaluate"] = task_instance["instruction_cn"]
                            elif lang == "en":
                                task_instance["prompt_to_evaluate"] = task_instance["instruction"]

                            tasks_to_run.append(task_instance)

    # Begin distributed processing
    tasks_for_this_process = tasks_to_run[accelerator.process_index::accelerator.num_processes]
    local_results = []

    # Display progress bar only on the main process
    progress_bar = tqdm(
        tasks_for_this_process,
        disable=not accelerator.is_main_process,
        desc="Overall Progress"
    )

    for sample in progress_bar:
        # Inference
        filename = sample["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)
        response = model.inference(instruction=sample["prompt_to_evaluate"], image_path=img_path)
        point = response["point"]
        try:
            print(f"[GPU {accelerator.process_index}] Processing image: {img_path}")
            tmp_img = Image.open(img_path)
            img_size = tmp_img.size
        except Exception as e:
            print(f"[GPU {accelerator.process_index}] Warning: Image not found at {img_path}, skipping.")
            continue
        sample["img_size"] = img_size
        point_in_pixel = [point[0] * img_size[0], point[1] * img_size[1]] if point else None

        sample_result = {
            "img_path": img_path, "group": sample.get("group"), "platform": sample.get("platform"),
            "application": sample.get("application"), "lang": sample.get("language"),
            "instruction_style": sample.get("instruction_style"), "prompt_to_evaluate": sample.get("prompt_to_evaluate"),
            "gt_type": sample.get("gt_type", 'positive'), "ui_type": sample.get("ui_type"),
            "task_filename": sample["task_filename"], "pred": point_in_pixel, "raw_response": response["raw_response"]
        }

        if sample["gt_type"] == "positive":
            correctness = eval_sample_positive_gt(
                sample, 
                response, 
                args.bbox,
                img_size,                 # 压缩后图像尺寸
                args.compression_ratio    # 新增参数
            )
            sample_result.update({"bbox": sample["bbox"]})
        else:
            correctness = eval_sample_negative_gt(sample, response)

        sample_result.update({"correctness": correctness})
        local_results.append(sample_result)

        # Evaluate and log intermediate results immediately
        current_metrics = calc_metric_for_result_list(local_results)
        print(
            f"[GPU {accelerator.process_index}] Sample: {sample['img_filename']:<30} | "
            f"Correctness: {correctness:<8} | "
            f"Running Acc for this GPU: {current_metrics['action_acc']:.4f} ({current_metrics['num_correct_action']}/{current_metrics['num_total']})"
        )
        # Write local results to the partial log file    
        with open(partial_log_path, 'w') as f:
            json.dump({"details": local_results}, f, indent=4)

    # wait for all processes to finish
    accelerator.wait_for_everyone()

    # 6. Combine partial results and evaluate (only in the main process)
    if accelerator.is_main_process:
        print("\nAll processes finished. Merging partial results into the final report...")

        all_details = []
        for i in range(accelerator.num_processes):
            # Create the partial log file path for each process
            part_file_path = os.path.join(
                log_dir,
                f"{os.path.splitext(log_filename)[0]}_part_{i}.json"
            )
            try:
                with open(part_file_path, 'r') as f:
                    data = json.load(f)
                    all_details.extend(data.get("details", []))
            except FileNotFoundError:
                print(f"Warning: Partial log file not found: {part_file_path}")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from partial log file: {part_file_path}")

        # Use the evaluate function to get the final report
        final_report = evaluate(all_details)
        with open(log_path, 'w') as f:
            json.dump(final_report, f, indent=4)

        print(f"\nFinal evaluation report saved to {log_path}")

        # delete partial log files
        print("Cleaning up partial log files...")
        for i in range(accelerator.num_processes):
            part_file_path = os.path.join(
                log_dir,
                f"{os.path.splitext(log_filename)[0]}_part_{i}.json"
            )
            if os.path.exists(part_file_path):
                os.remove(part_file_path)
        
if __name__ == "__main__":
    main()
