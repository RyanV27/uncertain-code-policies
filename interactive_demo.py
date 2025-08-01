import os
import pybullet
import pybullet_data
import numpy as np
import threading
import copy
import cv2
import subprocess
import pickle
from pathlib import Path
from datetime import datetime

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
from transformers import AutoTokenizer, AutoModelForCausalLM

# Importing user-defined fucntions and classes
from lmp import FunctionParser, var_exists, merge_dicts, exec_safe 
from tabletop_sim_env import Robotiq2F85, PickPlaceEnv
from lmp_utils import setup_LMP

from tabletop_config import ALL_BLOCKS, ALL_BOWLS, cfg_tabletop, model_name

# Name of the Hugging Face model repository
# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_id = "meta-llama/CodeLlama-13b-hf"

def main():
    print("Running Interactive_Demo.py!\n")
    
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

    # Initializing local Llama
    # To store the downloaded tokenizer and model in a specific folder add param "cache_dir=/path/to/folder/"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/scratch/rsvargh2/huggingface_models/")   
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir="/scratch/rsvargh2/huggingface_models/",    # To store the downloaded model in this folder
    )
    
    #  Initialize the Environment
    num_blocks = np.random.randint(0, 4) #@param {type:"slider", min:0, max:4, step:1}
    num_bowls = np.random.randint(0, 4) #@param {type:"slider", min:0, max:4, step:1}
    high_resolution = True #@param {type:"boolean"}
    high_frame_rate = False #@param {type:"boolean"}

    # setup env and LMP
    env = PickPlaceEnv(render=True, high_res=high_resolution, high_frame_rate=high_frame_rate)
    block_list = np.random.choice(ALL_BLOCKS, size=num_blocks, replace=False).tolist()
    bowl_list = np.random.choice(ALL_BOWLS, size=num_bowls, replace=False).tolist()
    obj_list = block_list + bowl_list
    _ = env.reset(obj_list)
    lmp_tabletop_ui = setup_LMP(env, cfg_tabletop, model_name=model_name, tokenizer=tokenizer, model=model)

    print("\nAvailable objects:")
    print(obj_list)

    # Creating the folder for storing the visualizations of the current run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  
    save_dir = Path(f"./runs/run_{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # display env
    # cv2.imshow('env', cv2.cvtColor(env.get_camera_image(env), cv2.COLOR_BGR2RGB))
    cv2.imwrite(save_dir / "env_img.jpg", cv2.cvtColor(env.get_camera_image(env), cv2.COLOR_RGB2BGR)) 

    # Interactive Demo
    while True:
        user_input = input("User input: ") #@param {allow-input: true, type:"string"}
        if user_input == '':
            break

        env.cache_video = []

        try:
            print("Running policy and recording video...")
            lmp_tabletop_ui(user_input, f'objects = {env.object_list}')

            # render video
            if env.cache_video:
                print(f"No. of frames: {len(env.cache_video)}")
        
                # Get frame properties
                height, width = env.cache_video[0].shape[:2]
                fps = 35 if high_frame_rate else 25
                
                # Define codec and output file
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID' or 'avc1'
                out = cv2.VideoWriter(save_dir / f"{user_input}.mp4", fourcc, fps, (width, height))
                
                for frame in env.cache_video:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
            
                out.release()
                print(f"Video saved as {save_dir}/{user_input}.mp4")
        except Exception as e:
            print(f"\nError:\n{e}\n")
            print("Exiting the simulation.")
            break

if __name__ == "__main__":
    main()