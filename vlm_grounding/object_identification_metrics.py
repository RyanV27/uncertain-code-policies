import argparse
import os
import json
from pathlib import Path

# ----------------- Do this if running the script inside test_code/------------------ #
import sys

# Adding parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)
# ---------------------------------------------------------------------------------- #

from tabletop_config import ALL_BLOCKS, ALL_BOWLS

def main():
    parser = argparse.ArgumentParser(description="Goes through all the sample environments with their actual and predicted objects and finds the metrcis for each possible object.")

    parser.add_argument("--provider", type=str, choices=["huggingface", "openai"], default="huggingface", help="Name of the model provider")
    parser.add_argument("--model", type=str, required=True, help="Name/ID of the VLM to test.")
    parser.add_argument(
        "--path",
        type=str,
        default="./runs/test_env",
        help=(
            "Path to the directory where environment images and a JSON",
            "file where an object list for each environment is stored."
        )
    )

    args = parser.parse_args()

    dir_path = Path(args.path)
    dir_path = dir_path / args.model if args.provider == "huggingface" else dir_path / args.provider / args.model
    object_lists_file_name = "vlm_env_obj_lists.json"
    scene_metrics_file_name = "individual_env_metrics.json"
    overall_metrics_file_name = "overall_metrics.json"
    object_lists_file_path = dir_path / object_lists_file_name
    scene_metrics_file_path = dir_path / scene_metrics_file_name
    overall_metrics_file_path = dir_path / overall_metrics_file_name

    if not os.path.exists(object_lists_file_path):
        print(f"{object_lists_file_path} does not exist.")
        print("Exiting")
        return

    # Initializing the dictionary of metrics for each object
    metrics_dict = {
        "env_id": 0,
        "TP": 0,
        "FP": 0,
        # "TN": 0,
        "FN": 0,
        # "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1_score": 0,
    }

    # Earlier when I tried to do for object level

    # object_metrics = {}
    # for obj in (ALL_BLOCKS + ALL_BOWLS):
    #     object_metrics[obj] = metrics_dict

    # for env in envs:
    #     act_list = sorted(env['actual'], key=(lambda x: (x.split()[1], x.split()[0])))
    #     pred_list = sorted(env['predicted'], key=(lambda x: (x.split()[1], x.split()[0])))

    #     first_bowl_id 
        
    #     for act_obj in act_list:


    # Quantitative Metrics at Scene level
    # Reading contents of the JSON file
    if os.path.exists(object_lists_file_path):
        with open(object_lists_file_path, "r") as f_in:
            envs = json.load(f_in)
    else:    
        print("JSON file cannot be found!")
        return

    scene_metrics = []
    total_TP = total_FP = total_FN = total_TN = 0
    total_precision = total_recall = num_correct_envs = 0
    
    for env in envs:
        if env["correct_format"] == False:
            continue

        num_correct_envs += 1
        
        scene_metrics.append(metrics_dict.copy())
        scene_metrics[-1]["env_id"] = env["id"]

        # print(f"Env id: {env['id']}")
        # print(f"Actual: {env['actual']}")
        # print(f"Predicted: {env['predicted']}")
        # print(f"Common (TP): {set(env['actual']) & set(env['predicted'])}, len = {len(set(env['actual']) & set(env['predicted']))}")
        # print(f"A - P (FN): {set(env['actual']) - set(env['predicted'])}, len = {len(set(env['actual']) - set(env['predicted']))}")
        # print(f"P - A (FP): {set(env['predicted']) - set(env['actual'])}, len = {len(set(env['predicted']) - set(env['actual']))}")
        # print(f"True Negative (TN): {set(ALL_BLOCKS + ALL_BOWLS) - set(env['actual'] + env['predicted'])}, len = {len(set(ALL_BLOCKS + ALL_BOWLS) - set(env['actual'] + env['predicted']))}\n")

        # Finding TP, FP, FN and TN
        scene_metrics[-1]["TP"] = len(set(env["actual"]) & set(env["predicted"]))     # Set intersection between actual and predicted list
        total_TP += scene_metrics[-1]["TP"]
        scene_metrics[-1]["FN"] = len(set(env["actual"]) - set(env["predicted"]))     # Set difference between actual and predicted list
        total_FN += scene_metrics[-1]["FN"]
        scene_metrics[-1]["FP"] = len(set(env["predicted"]) - set(env["actual"]))     # Set difference between predicted and actual list
        total_FP += scene_metrics[-1]["FP"]
        # scene_metrics[-1]["TN"] = len(set(ALL_BLOCKS + ALL_BOWLS) - set(env["actual"] + env["predicted"]))
        # total_TN += scene_metrics[-1]["TN"]

        # Calculating metrics for each scene
        # if len(env["actual"]) != 0:
        #     scene_metrics[-1]["accuracy"] = scene_metrics[-1]["TP"] / len(env["actual"])      # How many objects in the actual list got identified by VLM
        # else:
        #     scene_metrics[-1]["accuracy"] = 0
        
        if (scene_metrics[-1]["TP"] + scene_metrics[-1]["FP"]) != 0:
            scene_metrics[-1]["precision"] = scene_metrics[-1]["TP"] / (scene_metrics[-1]["TP"] + scene_metrics[-1]["FP"])
        else:
            scene_metrics[-1]["precision"] = 0
        total_precision += scene_metrics[-1]["precision"]

        if (scene_metrics[-1]["TP"] + scene_metrics[-1]["FN"]) != 0:
            scene_metrics[-1]["recall"] = scene_metrics[-1]["TP"] / (scene_metrics[-1]["TP"] + scene_metrics[-1]["FN"])
        else:
            scene_metrics[-1]["recall"] = 0
        total_recall += scene_metrics[-1]["recall"]
        
        if (scene_metrics[-1]["precision"] == 0) or (scene_metrics[-1]["recall"] == 0):
            scene_metrics[-1]["f1_score"] = 0
        else:
            scene_metrics[-1]["f1_score"] = (2 * scene_metrics[-1]["precision"] * scene_metrics[-1]["recall"]) / (scene_metrics[-1]["precision"] + scene_metrics[-1]["recall"])

        # print(f"Precision: {scene_metrics[-1]['precision']}")
        # print(f"Recall: {scene_metrics[-1]['recall']}")
        # print(f"F1 Score: {scene_metrics[-1]['f1_score']}\n\n")

    overall_metrics = {}
    
    # Micro scores
    overall_metrics["micro_precision"] = total_TP / (total_TP + total_FP)
    overall_metrics["micro_recall"]  = total_TP / (total_TP + total_FN)
    overall_metrics["micro_f1_score"] = (2 * overall_metrics["micro_precision"] * overall_metrics["micro_recall"]) / (overall_metrics["micro_precision"] + overall_metrics["micro_recall"])

    # print("Micro:-")
    # print(f"Precision: {overall_metrics['micro_precision']}")
    # print(f"Recall: {overall_metrics['micro_recall']}")
    # print(f"F1 Score: {overall_metrics['micro_f1_score']}\n")

    # Macro scores
    overall_metrics["macro_precision"] = total_precision / num_correct_envs
    overall_metrics["macro_recall"] = total_recall / num_correct_envs
    overall_metrics["macro_f1_score"] = (2 * overall_metrics["macro_precision"] * overall_metrics["macro_recall"]) / (overall_metrics["macro_precision"] + overall_metrics["macro_recall"])

    # print("Macro:-")
    # print(f"Precision: {overall_metrics['macro_precision']}")
    # print(f"Recall: {overall_metrics['macro_recall']}")
    # print(f"F1 Score: {overall_metrics['macro_f1_score']}\n")

    # Saving the individual environment metrics
    with open(scene_metrics_file_path, "w") as f_out:
        json.dump(scene_metrics, f_out)

    # Saving overall metrics
    with open(overall_metrics_file_path, "w") as f_out:
        json.dump(overall_metrics, f_out)

if __name__ == "__main__":
    main()