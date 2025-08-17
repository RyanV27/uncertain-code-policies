import os
import pybullet
import pybullet_data
import threading
import copy
import cv2
import subprocess
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image

# imports for LMPs
import shapely
import ast
import astunparse
import torch
from time import sleep
from shapely.geometry import *
from shapely.affinity import *
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

# ----------------- Do this if running the script inside test_code/------------------ #
import sys
import os

# Adding parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)
# ---------------------------------------------------------------------------------- #

# Importing user-defined fucntions and classes
from lmp import FunctionParser, var_exists, merge_dicts, exec_safe 
from tabletop_sim_env import Robotiq2F85, PickPlaceEnv
from lmp_utils import setup_LMP

from tabletop_config import ALL_BLOCKS, ALL_BOWLS, cfg_tabletop, model_name

def main():
    print("Running generate_test_envs.py!\n")
    
    parser = argparse.ArgumentParser(description="Generates a specified number of environments and saves their image and object list.")

    parser.add_argument("-n", type=int, required=True, help="Number of environments to generate.")
    parser.add_argument(
        "--path",
        type=str,
        default="./runs/test_env",
        help=(
            "Path to output directory where environment images and a JSON",
            "file where an object list for each environment is stored."
        )
    )
    parser.add_argument("--mode", choices=["w", "a"], default="w", help="Write mode to JSON file: 'w' = write (overwrite), 'a' = append")

    args = parser.parse_args()
                        
    # Download PyBullet assets.
    if not os.path.exists('ur5e/ur5e.urdf'):
        print('ur5e/ur5e.urdf doesn\'t exist')
        subprocess.run(['gdown', '--id', '1Cc_fDSBL6QiDvNT4dpfAEbhbALSVoWcc'], check=True)
        subprocess.run(['unzip', 'ur5e.zip'], check=True)

    if not os.path.exists('robotiq_2f_85/robotiq_2f_85.urdf'):  
        print('robotiq_2f_85/robotiq_2f_85.urdf doesn\'t exist')
        subprocess.run(['gdown', '--id', '1yOMEm-Zp_DL3nItG9RozPeJAmeOldekX'], check=True)
        subprocess.run(['unzip', 'robotiq_2f_85.zip'], check=True)

    if not os.path.exists('bowl/bowl.urdf'):
        print('bowl/bowl.urdf doesn\'t exist')  
        subprocess.run(['gdown', '--id', '1GsqNLhEl9dd4Mc3BM0dX3MibOI1FVWNM'], check=True)
        subprocess.run(['unzip', 'bowl.zip'], check=True)

    # Initialize the Environment
    high_resolution = True 
    high_frame_rate = False 
    env = PickPlaceEnv(render=True, high_res=high_resolution, high_frame_rate=high_frame_rate)

    # Creating the directory if it doesn't exist 
    save_dir = Path(args.path)
    save_dir.mkdir(parents=True, exist_ok=True)

    num_envs = args.n
    file_mode = args.mode

    start_id = 0
    
    if file_mode == 'a':
        try:
            # Listing all the json files and finding the last environment id
            json_files = [int(file.split('_')[1]) for file in os.listdir(save_dir) if file.endswith(".json")]
            last_env_file_id = max(json_files)
        
            with open(save_dir / f"env_{last_env_file_id}_obj_list.json", 'r') as f_in:
                data = json.load(f_in)

            start_id = data["id"] + 1
            print(f"Last env id: {start_id - 1}")
        except Exception as e:
            print("Failed to find or open last generated environment's json file.")
            print(f"Error: {e}")
    
    # Generating environments and prompting VLM
    for i in range(num_envs):
        print(f"\nEnv {start_id + i}")
        cur_env_data = {}
        
        num_blocks = np.random.randint(0, 5)   # originally min:0, max:4, step:1 
        num_bowls = np.random.randint(0, 5)    # originally min:0, max:4, step:1
        
        block_list = np.random.choice(ALL_BLOCKS, size=num_blocks, replace=False).tolist()
        bowl_list = np.random.choice(ALL_BOWLS, size=num_bowls, replace=False).tolist()
        obj_list = block_list + bowl_list
        
        _ = env.reset(obj_list)
        # lmp_tabletop_ui = setup_LMP(env, cfg_tabletop, model_name=model_name, tokenizer=tokenizer, model=model)

        print(f"Available objects: {obj_list}")
    
        cur_env_data['id'] = start_id + i
        cur_env_data['actual'] = obj_list
        
        # Saving env image with Pillow Image
        env_img = Image.fromarray(env.get_camera_image(env))
        env_img.save(save_dir / f"env_{start_id + i}_img.jpg")

        with open(save_dir / f"env_{start_id + i}_obj_list.json", "w") as f_out:
            json.dump(cur_env_data, f_out)

    print(f"Saved all the environment objects lists in \'{save_dir}\'")

if __name__ == "__main__":
    main()